param(
    [string]$Python = (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    [string]$VendorRoot = (Join-Path $PSScriptRoot 'third_party'),
    [string]$Version = '0.1.0'
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$vendor = (Resolve-Path -LiteralPath $VendorRoot).Path
$ffmpeg = Join-Path $vendor 'ffmpeg9\ffmpeg-9.0.1-essentials_build'
$model = Join-Path $vendor 'models\inpainting_lama_2025jan.onnx'
$distRoot = Join-Path $root "dist\$Version"
$destination = Join-Path $distRoot 'SubTitleRemover'
$archive = Join-Path $root "release\SubTitleRemover-$Version-win64.zip"
foreach ($required in @($Python, (Join-Path $ffmpeg 'bin\ffmpeg.exe'),
    (Join-Path $ffmpeg 'bin\ffprobe.exe'), (Join-Path $ffmpeg 'LICENSE'),
    (Join-Path $ffmpeg 'README.txt'), $model)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required build file is missing: $required"
    }
}
if ((Test-Path -LiteralPath $destination) -or (Test-Path -LiteralPath $archive)) {
    throw 'The release output already exists; choose a new version or archive it first.'
}

& $Python -m PyInstaller --onedir --name SubTitleRemover --collect-all rapidocr `
    --distpath $distRoot --workpath (Join-Path $root "build\$Version") `
    --specpath (Join-Path $root "build\$Version") (Join-Path $root 'remove_subtitles.py')
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }

$ffmpegTarget = Join-Path $destination 'third_party\ffmpeg9\ffmpeg-9.0.1-essentials_build'
$modelTarget = Join-Path $destination 'third_party\models'
New-Item -ItemType Directory -Path (Join-Path $ffmpegTarget 'bin'), $modelTarget -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $ffmpeg 'bin\ffmpeg.exe'), (Join-Path $ffmpeg 'bin\ffprobe.exe') -Destination (Join-Path $ffmpegTarget 'bin')
Copy-Item -LiteralPath (Join-Path $ffmpeg 'LICENSE'), (Join-Path $ffmpeg 'README.txt') -Destination $ffmpegTarget
Copy-Item -LiteralPath $model -Destination $modelTarget
Copy-Item -LiteralPath (Join-Path $root 'licenses\LaMa-LICENSE.txt') -Destination $modelTarget
Copy-Item -LiteralPath (Join-Path $root 'README.md'), (Join-Path $root 'THIRD_PARTY.md'), (Join-Path $root 'big_lama_worker.py') -Destination $destination
New-Item -ItemType Directory -Path (Split-Path -Parent $archive) -Force | Out-Null
Compress-Archive -LiteralPath $destination -DestinationPath $archive -CompressionLevel Optimal
Write-Output $archive
