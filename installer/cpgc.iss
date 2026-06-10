#define MyAppName    "CPGC"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "CPGC Project"
#define MyAppURL     "https://github.com/Union-Crax/CPGC"
#define MyAppExeName "cpgc-gui.exe"

[Setup]
AppId={{6F3A4B2E-8C1D-4E5F-9A0B-2D7E8F3C1A4B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\CPGC
DefaultGroupName=CPGC
AllowNoIcons=yes
OutputDir=installer\Output
OutputBaseFilename=CPGC-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Source tree root (two levels up from this .iss file inside installer\)
#define SourcePath "{#SourcePath}..\"

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon";     Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath";       Description: "Add install directory to user PATH"; GroupDescription: "Shell integration:"; Flags: unchecked
Name: "shellintegration"; Description: "Add right-click context menu (Compress / Open / Extract / Test)"; GroupDescription: "Shell integration:"

[Files]
Source: "{#SourcePath}target\release\cpgc-gui.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourcePath}target\release\cpgc.exe";     DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourcePath}README.md";                   DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\CPGC File Manager"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\CPGC File Manager"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; Optional: add install dir to user PATH
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{olddata};{app}"; \
  Check: NeedsAddPath(ExpandConstant('{app}')); Tasks: addtopath

[Run]
Filename: "{app}\cpgc.exe"; Parameters: "register"; \
  Flags: runhidden; Tasks: shellintegration; \
  StatusMsg: "Installing right-click menu..."
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName,' ','&')}}"; \
  Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\cpgc.exe"; Parameters: "unregister"; \
  Flags: runhidden; Tasks: shellintegration

[Code]
{ Returns True if Dir is not already in the PATH value. }
function NeedsAddPath(Dir: string): Boolean;
var
  OldPath: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', OldPath) then
  begin
    Result := True;
    Exit;
  end;
  Result := (Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(OldPath) + ';') = 0);
end;
