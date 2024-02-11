@echo off
setlocal enabledelayedexpansion

if not exist "%userprofile%\.kaggle\kaggle.json" (
  set /p USERNAME=Kaggle username: 
  echo.
  set /p APIKEY=Kaggle API key: 

  mkdir "%userprofile%\.kaggle"
  echo {"username":"!USERNAME!","key":"!APIKEY!"} > "%userprofile%\.kaggle\kaggle.json"
  attrib +R "%userprofile%\.kaggle\kaggle.json"
)

pip install kaggle --upgrade

kaggle competitions download -c carvana-image-masking-challenge -f train_hq.zip
powershell Expand-Archive train_hq.zip -DestinationPath data\imgs
move data\imgs\train_hq\* data\imgs\
rmdir /s /q data\imgs\train_hq
del /q train_hq.zip

kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip
powershell Expand-Archive train_masks.zip -DestinationPath data\masks
move data\masks\train_masks\* data\masks\
rmdir /s /q data\masks\train_masks
del /q train_masks.zip

exit /b 0
