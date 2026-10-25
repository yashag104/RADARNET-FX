@echo off
rem build_zcu102.bat — build the ZCU102 bitstream + hardware file on this PC.
rem Double-click it once; it runs unattended for about 30-60 minutes.
rem Produces rn_zcu102.bit, rn_zcu102.xsa and psu_init.tcl next to this file,
rem which run_zcu102.bat then uses. Needs a Vivado licence that covers ZU9EG.
cd /d "%~dp0"
set VIVADO=
if defined XILINX_VIVADO if exist "%XILINX_VIVADO%\bin\vivado.bat" set VIVADO=%XILINX_VIVADO%\bin\vivado.bat
if not defined VIVADO for %%V in (2026.2 2026.1 2025.2 2025.1 2024.2 2024.1 2023.2) do (
    if not defined VIVADO if exist "C:\Xilinx\Vivado\%%V\bin\vivado.bat" set VIVADO=C:\Xilinx\Vivado\%%V\bin\vivado.bat
    if not defined VIVADO if exist "C:\Xilinx\%%V\Vivado\bin\vivado.bat" set VIVADO=C:\Xilinx\%%V\Vivado\bin\vivado.bat
    if not defined VIVADO if exist "C:\AMDDesignTools\%%V\Vivado\bin\vivado.bat" set VIVADO=C:\AMDDesignTools\%%V\Vivado\bin\vivado.bat
    if not defined VIVADO if exist "D:\%%V\Vivado\bin\vivado.bat" set VIVADO=D:\%%V\Vivado\bin\vivado.bat
)
if not defined VIVADO set VIVADO=vivado
echo Using %VIVADO%
echo Building the ZCU102 hardware (W=16, 250 MHz). This takes 30-60 minutes...
echo Full log: %~dp0build\build.log
if not exist build mkdir build

call "%VIVADO%" -mode batch -nojournal -log build\build.log -source hw\build_zynq.tcl ^
     -tclargs 16 4 "%~dp0hw\mem_w16" "%~dp0build" 250 zcu102 1

if not exist build\rn_zed.xsa (
    echo.
    echo *** BUILD FAILED - send build\build.log to Claude.
    pause
    exit /b 1
)
for /r build\proj %%F in (rn_bd_wrapper.bit) do if exist "%%F" copy /y "%%F" rn_zcu102.bit >nul
copy /y build\rn_zed.xsa rn_zcu102.xsa >nul
copy /y build\rn_zed.xsa build\xsa.zip >nul
powershell -NoProfile -Command "Expand-Archive -Force 'build\xsa.zip' 'build\xsa'"
for /r build\xsa %%F in (psu_init.tcl) do if exist "%%F" copy /y "%%F" psu_init.tcl >nul

echo.
findstr /c:"SYSTEM_SUMMARY" build\build.log
if exist rn_zcu102.bit if exist psu_init.tcl (
    echo.
    echo BUILD OK - now double-click run_zcu102.bat
) else (
    echo *** Bitstream or psu_init.tcl missing - send build\build.log to Claude.
)
pause
