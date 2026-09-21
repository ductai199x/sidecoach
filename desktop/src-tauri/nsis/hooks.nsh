; Sidecoach installer hooks (Tauri NSIS, docs/app-plan.md section 2, "Delivery").
;
; The same setup file serves as "install, or just run": when the very version this
; installer carries is already installed for this user, the installer launches it and
; leaves instead of installing again. A different version installs over the old one
; (the updater's path too).
;
; The check runs before the first page shows (MUI2's GUI init, which the template lets
; a hooks file claim since it is included ahead of the pages), so a double-click on the
; setup file of an installed version opens the app, not an installer. A silent or
; passive run has no GUI init; the install section's hook catches it there.
;
; This file is included before the template's defines (PRODUCTNAME, VERSION,
; MAINBINARYNAME), so a function written here cannot use them: the registry key is
; named outright, the installed exe is read from the key's DisplayIcon, and the two
; versions compared are the file versions of that exe and of this installer.

!define SIDECOACH_UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Sidecoach"
!define MUI_CUSTOMFUNCTION_GUIINIT SidecoachRunInstalled

; Launch the installed copy and quit when it is this very version; else return.
Function SidecoachRunInstalled
  ClearErrors
  ReadRegStr $1 HKCU "${SIDECOACH_UNINSTKEY}" "DisplayIcon"
  ${If} ${Errors}
    Return
  ${EndIf}
  ; The template writes the path in quotes; without them the path is a path.
  StrCpy $2 $1 1
  ${If} $2 == '"'
    StrCpy $1 $1 -1 1
  ${EndIf}
  ${IfNot} ${FileExists} "$1"
    Return
  ${EndIf}
  ${GetFileVersion} "$1" $3
  ${GetFileVersion} "$EXEPATH" $4
  ${If} $3 != ""
  ${AndIf} $3 == $4
    ; Already here at this version: run it, and skip the install.
    Exec '"$1"'
    Quit
  ${EndIf}
FunctionEnd

!macro NSIS_HOOK_PREINSTALL
  Call SidecoachRunInstalled
!macroend

!macro NSIS_HOOK_POSTINSTALL
!macroend

!macro NSIS_HOOK_PREUNINSTALL
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
!macroend
