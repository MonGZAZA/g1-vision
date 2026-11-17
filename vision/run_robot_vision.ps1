# Launch Unitree G1 Vision over LAN (Front Camera + Livox MID-360 LiDAR UDP 56301)
param(
    [string]$CameraUrl = "unitree:http://192.168.123.100:8000/capture",
    [string]$CameraMount = "level",
    [int]$LidarPort = 56301,
    [int]$Port = 8107,
    [switch]$NoLidar
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

Write-Host "=== Unitree G1 Vision LAN + LiDAR Launcher ===" -ForegroundColor Cyan

# 1. Check Ethernet adapter & static IP 192.168.123.222/24
$eth = Get-NetAdapter -Name "Ethernet" -ErrorAction SilentlyContinue
if ($eth) {
    Write-Host "[net] Ethernet status: $($eth.Status)" -ForegroundColor Yellow
}
$ip = Get-NetIPAddress -InterfaceAlias "Ethernet" -AddressFamily IPv4 -ErrorAction SilentlyContinue
if (-not $ip -or $ip.IPAddress -ne "192.168.123.222") {
    Write-Host "[net] Static IP 192.168.123.222/24 not found on Ethernet. Attempting setup (requires Admin if not set)..." -ForegroundColor Yellow
    New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress "192.168.123.222" -PrefixLength 24 -ErrorAction SilentlyContinue | Out-Null
} else {
    Write-Host "[net] Ethernet IPv4: $($ip.IPAddress)/$($ip.PrefixLength) ($($ip.AddressState))" -ForegroundColor Green
}

# 2. Check G1 camera host reachability
$pingOk = Test-Connection -ComputerName "192.168.123.100" -Count 1 -Quiet -ErrorAction SilentlyContinue
if ($pingOk) {
    Write-Host "[net] G1 reachable at 192.168.123.100" -ForegroundColor Green
} else {
    Write-Host "[net] G1 (192.168.123.100) not reachable yet - service will auto-connect as soon as LAN cable is plugged in." -ForegroundColor Yellow
}

# 3. Check Windows Firewall rule for Livox MID-360 LiDAR UDP port
if (-not $NoLidar) {
    $ruleName = "G1 Vision Livox LiDAR UDP $LidarPort"
    $fw = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if (-not $fw) {
        try {
            New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol UDP -LocalPort $LidarPort -Action Allow -ErrorAction Stop | Out-Null
            Write-Host "[lidar] Added Windows Firewall Inbound Allow rule for UDP $LidarPort" -ForegroundColor Green
        } catch {
            Write-Host "[lidar] Tip: If LiDAR packets on UDP $LidarPort are blocked, run PowerShell as Admin once to allow UDP $LidarPort." -ForegroundColor Yellow
        }
    } else {
        Write-Host "[lidar] Firewall rule active for UDP $LidarPort" -ForegroundColor Green
    }
}

$argsList = @(
    "-m", "vision.service",
    "--camera", $CameraUrl,
    "--camera-mount", $CameraMount,
    "--hfov", "69",
    "--max-fps", "30",
    "--port", "$Port",
    "--joints-url", "none"
)
if (-not $NoLidar) {
    $argsList += @("--lidar-port", "$LidarPort")
}

Write-Host "[vision] Starting on http://127.0.0.1:$Port/ ..." -ForegroundColor Cyan
& python @argsList
