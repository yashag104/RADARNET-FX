@echo off
rem Double-click to program the ZCU102 and run the RadarNet-1D benchmark.
rem Needs: ZCU102 powered on, USB-JTAG (micro-USB J2) connected to this PC,
rem        boot-mode switch SW6 set to JTAG (all four ON).
rem Finds xsdb from XILINX_VITIS, then common install paths, then PATH.
cd /d "%~dp0"
set XSDB=
if defined XILINX_VITIS if exist "%XILINX_VITIS%\bin\xsdb.bat" set XSDB=%XILINX_VITIS%\bin\xsdb.bat
if not defined XSDB for %%V in (2026.2 2026.1 2025.2 2025.1 2024.2 2024.1 2023.2) do (
    if not defined XSDB if exist "C:\Xilinx\Vitis\%%V\bin\xsdb.bat" set XSDB=C:\Xilinx\Vitis\%%V\bin\xsdb.bat
    if not defined XSDB if exist "C:\Xilinx\%%V\Vitis\bin\xsdb.bat" set XSDB=C:\Xilinx\%%V\Vitis\bin\xsdb.bat
    if not defined XSDB if exist "C:\AMDDesignTools\%%V\Vitis\bin\xsdb.bat" set XSDB=C:\AMDDesignTools\%%V\Vitis\bin\xsdb.bat
    if not defined XSDB if exist "D:\%%V\Vitis\bin\xsdb.bat" set XSDB=D:\%%V\Vitis\bin\xsdb.bat
)
if not defined XSDB set XSDB=xsdb
echo Using %XSDB%
call "%XSDB%" run_zcu102.tcl
echo.
pause
