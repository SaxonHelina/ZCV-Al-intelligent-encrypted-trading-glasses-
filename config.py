{
    "server": {
        "host": "127.0.0.1",
        "port": 5000,
        "upload_endpoint": "/upload-image",
        "max_upload_size_mb": 10,
        "timeout_seconds": 30
    },
    "image": {
        "supported_formats": ["jpeg", "jpg", "png", "gif"],
        "max_width": 1920,
        "max_height": 1080,
        "compression_quality": 85,
        "image_save_dir": "uploaded_images",
        "image_filename_prefix": "device_"
    },
    "device": {
        "device_id": "wearable_001",
        "device_name": "Smart Glasses",
        "capture_interval_seconds": 5,
        "battery_level_alert_threshold": 20,
        "max_image_count_per_upload": 5
    },
    "network": {
        "use_proxy": false,
        "proxy_url": "http://proxy.example.com:8080",
        "proxy_username": "user",
        "proxy_password": "password",
        "network_timeout": 60
    },
    "security": {
        "ssl_enabled": true,
        "ssl_cert_file": "/path/to/cert.pem",
        "ssl_key_file": "/path/to/key.pem",
        "auth_token": "your-api-auth-token",
        "encryption_method": "AES-256",
        "encryption_key": "supersecretkey12345"
    },
    "logging": {
        "log_level": "INFO",
        "log_file": "device_log.log",
        "max_log_size_mb": 5,
        "log_backup_count": 3
    },
    "performance": {
        "image_compression_enabled": true,
        "image_compression_factor": 0.7,
        "upload_retry_attempts": 3,
        "retry_delay_seconds": 2
    },
    "notifications": {
        "email_alert_enabled": true,
        "email_recipient": "user@example.com",
        "sms_alert_enabled": false,
        "sms_recipient": "1234567890"
    },
    "health_monitoring": {
        "temperature_alert_enabled": true,
        "max_temperature_celsius": 40,
        "battery_drain_threshold_percent": 10
    },
    "backup": {
        "enable_auto_backup": true,
        "backup_dir": "backup/",
        "backup_frequency_hours": 24,
        "cloud_backup_enabled": false,
        "cloud_backup_service_url": "https://backupservice.example.com"
    }
}
