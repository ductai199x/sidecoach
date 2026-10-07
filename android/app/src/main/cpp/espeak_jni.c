/*
 * The four espeak-ng calls phonemizer makes (backend/espeak/api.py), for the phone's port of
 * it (com.sidecoach.voice.Espeak): initialise, list the voices, choose one, and turn text into
 * phonemes clause by clause. Strings cross as UTF-8 bytes, decoded on the Kotlin side, so no
 * character is lost to the JNI's modified UTF-8.
 *
 * espeak-ng keeps its state in globals; the Kotlin side calls it under one lock, as
 * kokoro-onnx does with `_espeak_lock`.
 */

#include <jni.h>
#include <stdlib.h>
#include <string.h>

#include <espeak-ng/speak_lib.h>

static jbyteArray bytes_of(JNIEnv *env, const char *text) {
    jsize n = (jsize) strlen(text);
    jbyteArray out = (*env)->NewByteArray(env, n);
    if (out != NULL && n > 0) (*env)->SetByteArrayRegion(env, out, 0, n, (const jbyte *) text);
    return out;
}

static char *string_of(JNIEnv *env, jbyteArray utf8) {
    jsize n = (*env)->GetArrayLength(env, utf8);
    char *text = malloc((size_t) n + 1);
    if (text == NULL) return NULL;
    (*env)->GetByteArrayRegion(env, utf8, 0, n, (jbyte *) text);
    text[n] = 0;
    return text;
}

/* espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_path, 0): the sample rate, or <= 0. */
JNIEXPORT jint JNICALL
Java_com_sidecoach_voice_Espeak_initialize(JNIEnv *env, jclass cls, jbyteArray data_path) {
    (void) cls;
    char *path = string_of(env, data_path);
    if (path == NULL) return -1;
    int rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, path, 0);
    free(path);
    return rate;
}

/* espeak_Info: the version string. */
JNIEXPORT jbyteArray JNICALL
Java_com_sidecoach_voice_Espeak_info(JNIEnv *env, jclass cls) {
    (void) cls;
    const char *data = NULL;
    const char *version = espeak_Info(&data);
    return bytes_of(env, version != NULL ? version : "");
}

/*
 * espeak_ListVoices(NULL), as phonemizer reads each entry: the name, the first language after
 * its priority byte, and the identifier -- three byte strings per voice, in the listed order.
 */
JNIEXPORT jobjectArray JNICALL
Java_com_sidecoach_voice_Espeak_listVoices(JNIEnv *env, jclass cls) {
    (void) cls;
    const espeak_VOICE **voices = espeak_ListVoices(NULL);
    jsize count = 0;
    while (voices != NULL && voices[count] != NULL) count++;
    jclass bytes = (*env)->FindClass(env, "[B");
    jobjectArray out = (*env)->NewObjectArray(env, count * 3, bytes, NULL);
    for (jsize i = 0; i < count; i++) {
        const espeak_VOICE *voice = voices[i];
        const char *language = voice->languages != NULL && voice->languages[0] != 0 ? voice->languages + 1 : "";
        jbyteArray name = bytes_of(env, voice->name != NULL ? voice->name : "");
        jbyteArray lang = bytes_of(env, language);
        jbyteArray ident = bytes_of(env, voice->identifier != NULL ? voice->identifier : "");
        (*env)->SetObjectArrayElement(env, out, i * 3, name);
        (*env)->SetObjectArrayElement(env, out, i * 3 + 1, lang);
        (*env)->SetObjectArrayElement(env, out, i * 3 + 2, ident);
        (*env)->DeleteLocalRef(env, name);
        (*env)->DeleteLocalRef(env, lang);
        (*env)->DeleteLocalRef(env, ident);
    }
    return out;
}

/* espeak_SetVoiceByName: 0 on success. */
JNIEXPORT jint JNICALL
Java_com_sidecoach_voice_Espeak_setVoiceByName(JNIEnv *env, jclass cls, jbyteArray name) {
    (void) cls;
    char *text = string_of(env, name);
    if (text == NULL) return -1;
    int status = espeak_SetVoiceByName(text);
    free(text);
    return status;
}

/* espeak_GetCurrentVoice: the current voice's name, empty when none is set. */
JNIEXPORT jbyteArray JNICALL
Java_com_sidecoach_voice_Espeak_currentVoiceName(JNIEnv *env, jclass cls) {
    (void) cls;
    espeak_VOICE *voice = espeak_GetCurrentVoice();
    return bytes_of(env, voice != NULL && voice->name != NULL ? voice->name : "");
}

/*
 * espeak_TextToPhonemes called until the text is used up, as EspeakWrapper.text_to_phonemes
 * does: one result per clause, the empty ones dropped, the rest joined by single spaces.
 */
JNIEXPORT jbyteArray JNICALL
Java_com_sidecoach_voice_Espeak_textToPhonemes(
    JNIEnv *env, jclass cls, jbyteArray utf8, jint text_mode, jint phonemes_mode) {
    (void) cls;
    char *text = string_of(env, utf8);
    if (text == NULL) return bytes_of(env, "");
    size_t size = 0, capacity = 256;
    char *joined = malloc(capacity);
    if (joined == NULL) {
        free(text);
        return bytes_of(env, "");
    }
    joined[0] = 0;
    const void *pointer = text;
    while (pointer != NULL) {
        const char *phonemes = espeak_TextToPhonemes(&pointer, text_mode, phonemes_mode);
        if (phonemes == NULL || phonemes[0] == 0) continue;
        size_t n = strlen(phonemes);
        size_t need = size + (size > 0 ? 1 : 0) + n + 1;
        if (need > capacity) {
            while (capacity < need) capacity *= 2;
            char *grown = realloc(joined, capacity);
            if (grown == NULL) break;
            joined = grown;
        }
        if (size > 0) joined[size++] = ' ';
        memcpy(joined + size, phonemes, n);
        size += n;
        joined[size] = 0;
    }
    free(text);
    jbyteArray out = bytes_of(env, joined);
    free(joined);
    return out;
}
