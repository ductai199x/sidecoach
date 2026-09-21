// The voices' labels, from their Kokoro names: "af_sarah" is American, female, Sarah.
// The list itself comes from the sidecar (the English voices its model file holds).

const ACCENT: Record<string, string> = { a: "American", b: "British" };
const SEX: Record<string, string> = { f: "Female", m: "Male" };

/** "af_sarah" -> "American Female, Sarah"; a name outside the scheme is shown as it is. */
export function voiceLabel(name: string): string {
  const accent = ACCENT[name[0]], sex = SEX[name[1]];
  if (!accent || !sex || name[2] !== "_" || name.length < 4) return name;
  const who = name.slice(3);
  return `${accent} ${sex}, ${who[0].toUpperCase()}${who.slice(1)}`;
}
