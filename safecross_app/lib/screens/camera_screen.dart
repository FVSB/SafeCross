import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/decision_result.dart';
import '../services/yolo_inference_service.dart';

/// Intervalo mínimo entre análisis de frames.
const _kAnalysisInterval = Duration(seconds: 3);

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraScreen({super.key, required this.cameras});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _cameraReady = false;
  bool _analyzing = false;
  bool _modelsLoading = true;

  DecisionResult? _lastResult;
  String? _errorMessage;
  DecisionState? _lastSpokenState;

  late FlutterTts _tts;
  final YoloInferenceService _inferenceService = YoloInferenceService();

  DateTime? _lastAnalysisTime;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initTts();
    _loadModelsAndStart();
  }

  // ── TTS ────────────────────────────────────────────────────────────────────

  void _initTts() {
    _tts = FlutterTts();
    _tts.setLanguage('es-ES');
    _tts.setSpeechRate(0.5);
    _tts.setVolume(1.0);
  }

  Future<void> _speak(String text) async {
    await _tts.stop();
    await _tts.speak(text);
  }

  // ── Inicialización ─────────────────────────────────────────────────────────

  Future<void> _loadModelsAndStart() async {
    // Cargar modelos TFLite en segundo plano
    try {
      await _inferenceService.loadModels();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _modelsLoading = false;
        _errorMessage =
            'Error al cargar modelos: $e\n\n'
            'Asegúrate de haber ejecutado scripts/export_tflite.py '
            'y copiado los .tflite a assets/models/';
      });
      return;
    }

    if (!mounted) return;
    setState(() => _modelsLoading = false);

    // Pedir permiso de cámara e iniciar
    await _requestCameraAndStart();
  }

  // ── Cámara ─────────────────────────────────────────────────────────────────

  Future<void> _requestCameraAndStart() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _errorMessage = 'Se necesita permiso de cámara.');
      return;
    }
    if (widget.cameras.isEmpty) {
      setState(() =>
          _errorMessage = 'No se encontraron cámaras en este dispositivo.');
      return;
    }
    await _startCamera(widget.cameras.first);
  }

  Future<void> _startCamera(CameraDescription camera) async {
    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      // YUV420 es el formato requerido por flutter_vision en Android
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await controller.initialize();
    } catch (e) {
      setState(() => _errorMessage = 'Error al iniciar cámara: $e');
      return;
    }

    if (!mounted) return;

    setState(() {
      _controller = controller;
      _cameraReady = true;
    });

    // Iniciar stream de frames para análisis continuo
    await controller.startImageStream(_onFrameAvailable);
  }

  /// Recibe cada frame del sensor y decide si analizarlo según el intervalo.
  void _onFrameAvailable(CameraImage image) {
    if (_analyzing) return;

    final now = DateTime.now();
    if (_lastAnalysisTime != null &&
        now.difference(_lastAnalysisTime!) < _kAnalysisInterval) {
      return;
    }
    _lastAnalysisTime = now;
    _analyzeFrame(image);
  }

  Future<void> _analyzeFrame(CameraImage image) async {
    setState(() => _analyzing = true);

    try {
      // Rotación 90° para corregir la orientación en portrait
      final result = await _inferenceService.decide(image, rotation: 90);

      if (!mounted) return;
      setState(() {
        _lastResult = result;
        _errorMessage = null;
      });

      // Hablar solo si el estado cambió
      if (result.state != _lastSpokenState) {
        _lastSpokenState = result.state;
        await _speak(result.speechText);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _errorMessage = 'Error de inferencia: $e');
    } finally {
      if (mounted) setState(() => _analyzing = false);
    }
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;

    if (state == AppLifecycleState.inactive) {
      controller.stopImageStream();
      controller.dispose();
      setState(() => _cameraReady = false);
    } else if (state == AppLifecycleState.resumed) {
      _startCamera(controller.description);
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.stopImageStream();
    _controller?.dispose();
    _inferenceService.dispose();
    _tts.stop();
    super.dispose();
  }

  // ── UI ─────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Vista de cámara
          if (_cameraReady && _controller != null)
            CameraPreview(_controller!)
          else
            _buildPlaceholder(),

          // Panel de decisión (parte inferior)
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: _buildDecisionPanel(),
          ),

          // Spinner de análisis (esquina superior derecha)
          if (_analyzing)
            const Positioned(
              top: 50,
              right: 16,
              child: _AnalyzingBadge(),
            ),
        ],
      ),
    );
  }

  Widget _buildPlaceholder() {
    if (_modelsLoading) {
      return const _LoadingScreen(message: 'Cargando modelos de IA…');
    }
    return Container(
      color: Colors.black,
      child: Center(
        child: _errorMessage != null
            ? Padding(
                padding: const EdgeInsets.all(32),
                child: Text(
                  _errorMessage!,
                  style:
                      const TextStyle(color: Colors.white70, fontSize: 15),
                  textAlign: TextAlign.center,
                ),
              )
            : const CircularProgressIndicator(color: Colors.white),
      ),
    );
  }

  Widget _buildDecisionPanel() {
    if (_modelsLoading) return const SizedBox.shrink();

    if (_errorMessage != null && _lastResult == null) {
      return _DecisionPanel(
        color: Colors.grey[900]!,
        icon: Icons.error_outline,
        label: 'ERROR',
        detail: _errorMessage!,
      );
    }

    if (_lastResult == null) {
      return _DecisionPanel(
        color: Colors.grey[850]!,
        icon: Icons.hourglass_empty,
        label: 'ANALIZANDO…',
        detail: 'Apunta la cámara hacia el cruce peatonal',
      );
    }

    final result = _lastResult!;
    return _DecisionPanel(
      color: result.backgroundColor,
      icon: result.icon,
      label: result.mainLabel,
      detail: result.reason,
    );
  }
}

// ── Widgets auxiliares ────────────────────────────────────────────────────────

class _LoadingScreen extends StatelessWidget {
  final String message;
  const _LoadingScreen({required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.black,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const CircularProgressIndicator(color: Colors.white),
          const SizedBox(height: 24),
          Text(
            message,
            style: const TextStyle(color: Colors.white70, fontSize: 16),
          ),
        ],
      ),
    );
  }
}

class _DecisionPanel extends StatelessWidget {
  final Color color;
  final IconData icon;
  final String label;
  final String detail;

  const _DecisionPanel({
    required this.color,
    required this.icon,
    required this.label,
    required this.detail,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 400),
      color: color.withAlpha(230),
      padding: const EdgeInsets.fromLTRB(24, 20, 24, 36),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Icon(icon, size: 56, color: Colors.white),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  label,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 26,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.2,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  detail,
                  style:
                      const TextStyle(color: Colors.white70, fontSize: 13),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _AnalyzingBadge extends StatelessWidget {
  const _AnalyzingBadge();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(20),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 14,
            height: 14,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: Colors.white,
            ),
          ),
          SizedBox(width: 6),
          Text(
            'Analizando',
            style: TextStyle(color: Colors.white, fontSize: 12),
          ),
        ],
      ),
    );
  }
}
