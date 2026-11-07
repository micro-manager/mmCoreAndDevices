# Copyright (c) 2026, Dr. Volodymyr Cherkas
# SPDX-License-Identifier: BSD-3-Clause

param([string]$AdapterPath, [string]$CameraIP = '192.168.206.206', [string]$MicroManagerPath = 'C:\Program Files\Micro-Manager-2.0', [switch]$Probe, [switch]$Capabilities, [switch]$ConfigCheck, [switch]$Persistence)
$ErrorActionPreference = 'Stop'
if (!$AdapterPath) {
    $AdapterPath = Join-Path $PSScriptRoot '..\binaries\windows-x64'
}
$env:EMERGENT_ADAPTER_PATH = $AdapterPath
$env:EMERGENT_CAMERA_IP = $CameraIP
$env:EMERGENT_TEST_OUTPUT = $PSScriptRoot
$env:PATH = 'C:\Program Files\EVT\eSDK\bin;C:\Program Files\EVT\eSDK\genicam\bin\Win64_x64;C:\Program Files\EVT\eSDK\ffmpeg\bin;' + $env:PATH
if ($Persistence) {
    & "$MicroManagerPath\jre\bin\java.exe" "-Djava.library.path=$MicroManagerPath" -cp "$MicroManagerPath\plugins\Micro-Manager\*" bsh.Interpreter "$PSScriptRoot\persistence-tests.bsh"
    if (!(Select-String -LiteralPath "$PSScriptRoot\persistence-results.txt" -Pattern '^RESULT: failures=0$' -Quiet)) { throw 'Settings persistence tests failed' }
    exit $LASTEXITCODE
}
if ($ConfigCheck) {
    & "$MicroManagerPath\jre\bin\java.exe" "-Djava.library.path=$MicroManagerPath" -cp "$MicroManagerPath\plugins\Micro-Manager\*" bsh.Interpreter "$PSScriptRoot\check-config.bsh"
    exit $LASTEXITCODE
}
if ($Capabilities) {
    & 'C:\Program Files\Python312\python.exe' "$PSScriptRoot\camera-capabilities.py"
    exit $LASTEXITCODE
}
if ($Probe) {
    & 'C:\Program Files\Python312\python.exe' "$PSScriptRoot\sdk-probe.py"
    exit $LASTEXITCODE
}
& "$MicroManagerPath\jre\bin\java.exe" "-Djava.library.path=$MicroManagerPath" -cp "$MicroManagerPath\plugins\Micro-Manager\*" bsh.Interpreter "$PSScriptRoot\functional-tests.bsh"
if ($LASTEXITCODE -ne 0) { throw "Java exited with $LASTEXITCODE" }
if (!(Select-String -LiteralPath "$PSScriptRoot\functional-results.txt" -Pattern '^RESULT: failures=0$' -Quiet)) { throw 'Camera functional tests failed; inspect functional-results.txt' }
