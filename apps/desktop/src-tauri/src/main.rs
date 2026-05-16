use base64::{engine::general_purpose, Engine as _};
use image::{codecs::jpeg::JpegEncoder, DynamicImage};
use serde::Serialize;

#[derive(Serialize)]
struct CaptureResponse {
    base64: String,
    width: u32,
    height: u32,
    monitor_id: String,
}

const CAPTURE_MAX_SIDE: u32 = 1280;
const JPEG_QUALITY: u8 = 62;

#[tauri::command]
fn capture_screen_jpeg() -> Result<CaptureResponse, String> {
    let monitors = xcap::Monitor::all().map_err(|error| error.to_string())?;
    let monitor = monitors
        .first()
        .ok_or_else(|| "No monitor available for capture".to_string())?;
    let image = monitor.capture_image().map_err(|error| error.to_string())?;
    let mut dynamic = DynamicImage::ImageRgba8(image);
    let side = dynamic.width().max(dynamic.height());
    if side > CAPTURE_MAX_SIDE {
        let scale = CAPTURE_MAX_SIDE as f32 / side as f32;
        let width = (dynamic.width() as f32 * scale).round() as u32;
        let height = (dynamic.height() as f32 * scale).round() as u32;
        dynamic = dynamic.resize(width, height, image::imageops::FilterType::Triangle);
    }
    let width = dynamic.width();
    let height = dynamic.height();
    let mut bytes = Vec::new();
    JpegEncoder::new_with_quality(&mut bytes, JPEG_QUALITY)
        .encode_image(&dynamic.to_rgb8())
        .map_err(|error| error.to_string())?;
    Ok(CaptureResponse {
        base64: general_purpose::STANDARD.encode(bytes),
        width,
        height,
        monitor_id: monitor.name().unwrap_or_else(|| "primary".to_string()),
    })
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![capture_screen_jpeg])
        .run(tauri::generate_context!())
        .expect("failed to run Lumina Learn desktop app");
}
