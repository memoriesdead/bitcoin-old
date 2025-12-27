# TURBO DOWNLOAD - PowerShell Maximum Speed
# Uses BITS for maximum bandwidth utilization

$ErrorActionPreference = "Continue"

Write-Host "=" * 70
Write-Host "TURBO DOWNLOAD - MAXIMUM SPEED MODE" -ForegroundColor Cyan
Write-Host "=" * 70

$dataDir = "C:\Users\kevin\livetrading\data"
New-Item -ItemType Directory -Force -Path "$dataDir\orbitaal" | Out-Null

# Exchange Addresses (1GB)
$exchangeUrl = "https://drive.switch.ch/index.php/s/ag4OnNgwf7LhWFu/download"
$exchangeFile = "$dataDir\entity_addresses.zip"

# ORBITAAL Files from Zenodo
$zenodoApi = "https://zenodo.org/api/records/12581515"

Write-Host "`n[1] DOWNLOADING EXCHANGE ADDRESSES (1 GB)" -ForegroundColor Yellow
if (-not (Test-Path $exchangeFile)) {
    Write-Host "    Starting BITS transfer..."
    Start-BitsTransfer -Source $exchangeUrl -Destination $exchangeFile -Priority High -DisplayName "Exchange Addresses"
    Write-Host "    [OK] Downloaded" -ForegroundColor Green
} else {
    Write-Host "    [SKIP] Already exists" -ForegroundColor Gray
}

# Extract if needed
$extractDir = "$dataDir\entity_addresses"
if ((Test-Path $exchangeFile) -and (-not (Test-Path $extractDir))) {
    Write-Host "    Extracting..."
    Expand-Archive -Path $exchangeFile -DestinationPath $extractDir -Force
    Write-Host "    [OK] Extracted" -ForegroundColor Green
}

Write-Host "`n[2] DOWNLOADING ORBITAAL (156 GB)" -ForegroundColor Yellow
Write-Host "    Fetching file list from Zenodo..."

try {
    $response = Invoke-RestMethod -Uri $zenodoApi -TimeoutSec 60
    $files = $response.files

    Write-Host "    Found $($files.Count) files"

    # Priority files first
    $priorityFiles = @(
        "orbitaal-stream_graph.tar.gz",
        "orbitaal-nodetable.tar.gz"
    )

    foreach ($fileName in $priorityFiles) {
        $file = $files | Where-Object { $_.key -eq $fileName }
        if ($file) {
            $dest = "$dataDir\orbitaal\$($file.key)"
            $sizeGB = [math]::Round($file.size / 1GB, 2)

            if (Test-Path $dest) {
                $existingSize = (Get-Item $dest).Length
                if ($existingSize -ge ($file.size * 0.99)) {
                    Write-Host "    [SKIP] $fileName (already exists)" -ForegroundColor Gray
                    continue
                }
            }

            Write-Host "    Downloading $fileName ($sizeGB GB)..." -ForegroundColor Cyan

            # Use BITS for large files (supports resume)
            $job = Start-BitsTransfer -Source $file.links.self -Destination $dest -Asynchronous -Priority High -DisplayName $fileName

            # Monitor progress
            while (($job.JobState -eq "Transferring") -or ($job.JobState -eq "Connecting")) {
                $pct = [math]::Round(($job.BytesTransferred / $job.BytesTotal) * 100, 1)
                $transferred = [math]::Round($job.BytesTransferred / 1MB, 1)
                $total = [math]::Round($job.BytesTotal / 1MB, 1)
                Write-Host "`r    [$pct%] $transferred / $total MB    " -NoNewline
                Start-Sleep -Seconds 2
            }

            if ($job.JobState -eq "Transferred") {
                Complete-BitsTransfer -BitsJob $job
                Write-Host "`n    [OK] $fileName" -ForegroundColor Green
            } else {
                Write-Host "`n    [FAIL] $fileName - $($job.JobState)" -ForegroundColor Red
            }
        }
    }

} catch {
    Write-Host "    [ERROR] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n" + "=" * 70
Write-Host "DOWNLOAD COMPLETE" -ForegroundColor Green
Write-Host "=" * 70
