; Sidecoach installer hooks (Tauri NSIS, docs/app-plan.md section 2, "Delivery").
;
; The same setup file serves as "install, or just run": when the very version this
; installer carries is already installed for this user, the installer launches it and
; leaves instead of installing again. A different version installs over the old one
; (the updater's path too). The registry key is the one Tauri's NSIS template writes
; for a per-user install.

!macro NSIS_HOOK_PREINSTALL
  ClearErrors
  ReadRegStr $0 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCTNAME}" "DisplayVersion"
  ReadRegStr $1 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCTNAME}" "InstallLocation"
  ${If} $0 == "${VERSION}"
  ${AndIf} ${FileExists} "$1\${MAINBINARYNAME}.exe"
    ; Already here at this version: run it, and skip the install.
    Exec '"$1\${MAINBINARYNAME}.exe"'
    Quit
  ${EndIf}
!macroend

!macro NSIS_HOOK_POSTINSTALL
!macroend

!macro NSIS_HOOK_PREUNINSTALL
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
!macroend
