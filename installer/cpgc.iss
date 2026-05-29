; Inno Setup script for the CPGC Windows installer.
; Built in CI (see .github/workflows/build.yml). Produces dist\CPGC-Setup.exe,
; a real installer that adds Start-Menu / desktop shortcuts for the GUI and
; bundles the CLI alongside it.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

[Setup]
AppName=CPGC
AppVersion={#MyAppVersion}
AppPublisher=CPGC
DefaultDirName={autopf}\CPGC
DefaultGroupName=CPGC
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=CPGC-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"
Name: "addtopath"; Description: "Add the CLI (cpgc) to PATH"; GroupDescription: "Command line:"

[Files]
Source: "target\release\cpgc-gui.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "target\release\cpgc.exe";     DestDir: "{app}"; Flags: ignoreversion
Source: "README.md";                   DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\CPGC File Manager"; Filename: "{app}\cpgc-gui.exe"
Name: "{group}\Uninstall CPGC";    Filename: "{uninstallexe}"
Name: "{autodesktop}\CPGC File Manager"; Filename: "{app}\cpgc-gui.exe"; Tasks: desktopicon

[Registry]
; Optionally add the install dir to the user PATH for the CLI.
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{olddata};{app}"; Tasks: addtopath; \
  Check: NeedsAddPath('{app}')

[Run]
Filename: "{app}\cpgc-gui.exe"; Description: "Launch CPGC File Manager"; \
  Flags: nowait postinstall skipifsilent

[Code]
function NeedsAddPath(Param: string): Boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + ExpandConstant(Param) + ';', ';' + OrigPath + ';') = 0;
end;
