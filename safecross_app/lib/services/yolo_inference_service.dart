import 'package:camera/camera.dart';
import 'package:flutter_vision/flutter_vision.dart';

import '../models/decision_result.dart';

/// Confianza mínima para aceptar una detección.
const _kConf = 0.50;

/// Nombre de la clase verde en el modelo de semáforos (debe coincidir con
/// la línea en assets/labels/lights.txt).
const _kGreenClass = 'verde';

/// Servicio de inferencia completamente local (sin internet).
///
/// Carga dos modelos TFLite desde los assets de la app:
///   - crosswalks.tflite — detecta pasos de peatones
///   - lights.tflite     — detecta el color del semáforo
///
/// Uso:
///   final service = YoloInferenceService();
///   await service.loadModels();
///   final result = await service.decide(cameraImage);
///   await service.dispose();
class YoloInferenceService {
  final FlutterVision _crosswalkVision = FlutterVision();
  final FlutterVision _lightsVision = FlutterVision();

  bool _loaded = false;

  bool get isLoaded => _loaded;

  /// Carga los dos modelos TFLite desde los assets.
  /// Lanza [Exception] si los archivos no están presentes.
  Future<void> loadModels() async {
    await _crosswalkVision.loadYoloModel(
      labels: 'assets/labels/crosswalks.txt',
      modelPath: 'assets/models/crosswalks.tflite',
      modelVersion: 'yolov8',
      quantization: false,
      numThreads: 2,
      useGpu: false,
    );

    await _lightsVision.loadYoloModel(
      labels: 'assets/labels/lights.txt',
      modelPath: 'assets/models/lights.tflite',
      modelVersion: 'yolov8',
      quantization: false,
      numThreads: 2,
      useGpu: false,
    );

    _loaded = true;
  }

  /// Ejecuta la lógica de decisión sobre un frame de la cámara.
  ///
  /// [frame] es un [CameraImage] en formato YUV420 (Android) o BGRA (iOS).
  /// [rotation] es la rotación en grados necesaria para corregir la
  /// orientación del sensor (normalmente 90° en portrait).
  Future<DecisionResult> decide(CameraImage frame, {int rotation = 90}) async {
    assert(_loaded, 'Llama a loadModels() primero');

    final bytesList = frame.planes.map((p) => p.bytes).toList();
    final h = frame.height;
    final w = frame.width;

    // ── 1. Detección de paso de peatones ──────────────────────────────────
    final crosswalkDets = await _crosswalkVision.yoloOnFrame(
      bytesList: bytesList,
      imageHeight: h,
      imageWidth: w,
      iouThreshold: 0.4,
      confThreshold: _kConf,
      classThreshold: _kConf,
      rotation: rotation,
    );

    if (crosswalkDets.isEmpty) {
      return DecisionResult(
        canCross: false,
        crosswalkDetected: false,
        nCrosswalks: 0,
        lightDetected: false,
        lightColor: null,
        reason: 'No se detectó un paso de peatones en la imagen.',
      );
    }

    // ── 2. Detección del semáforo ──────────────────────────────────────────
    final lightDets = await _lightsVision.yoloOnFrame(
      bytesList: bytesList,
      imageHeight: h,
      imageWidth: w,
      iouThreshold: 0.4,
      confThreshold: _kConf,
      classThreshold: _kConf,
      rotation: rotation,
    );

    if (lightDets.length != 1) {
      return DecisionResult(
        canCross: false,
        crosswalkDetected: true,
        nCrosswalks: crosswalkDets.length,
        lightDetected: false,
        lightColor: null,
        reason:
            'Se esperaba 1 semáforo, se detectaron ${lightDets.length}.',
      );
    }

    final lightColor = lightDets.first['tag'] as String;

    if (lightColor != _kGreenClass) {
      return DecisionResult(
        canCross: false,
        crosswalkDetected: true,
        nCrosswalks: crosswalkDets.length,
        lightDetected: true,
        lightColor: lightColor,
        reason: "El semáforo está en '$lightColor', no en verde.",
      );
    }

    // ── 3. ¡Cruce seguro! ─────────────────────────────────────────────────
    return DecisionResult(
      canCross: true,
      crosswalkDetected: true,
      nCrosswalks: crosswalkDets.length,
      lightDetected: true,
      lightColor: _kGreenClass,
      reason: 'Paso de peatones visible y semáforo en verde. Puede cruzar.',
    );
  }

  /// Libera los recursos de los modelos.
  Future<void> dispose() async {
    await _crosswalkVision.closeYoloModel();
    await _lightsVision.closeYoloModel();
    _loaded = false;
  }
}
