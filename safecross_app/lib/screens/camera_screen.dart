import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/decision_result.dart';
import '../services/safecross_service.dart';

/// Intervalo entre análisis de frames (segundos).
const _kAnalysisInterval = Duration(seconds: 3);

/// URL del servidor por defecto (cambiar según red local).
const _kDefaultServerUrl = 'http://192.168.1.100:8000';

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

  DecisionResult? _lastResult;
  String? _errorMessage;

  late FlutterTts _tts;
  late String _serverUrl;
  SafeCrossService? _service;

  Timer? _analysisTimer;
  DecisionState? _lastSpokenState;

  // Controlador del TextField en el diálogo de ajustes
  final _urlController = TextEditingController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initTts();
    _loadSettings().then((_) => _requestCameraAndStart());
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

  // ── Ajustes ────────────────────────────────────────────────────────────────

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    _serverUrl = prefs.getString('server_url') ?? _kDefaultServerUrl;
    _service = SafeCrossService(baseUrl: _serverUrl);
  }

  Future<void> _saveSettings(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('server_url', url);
    setState(() {
      _serverUrl = url;
      _service = SafeCrossService(baseUrl: url);
    });
  }

  // ── Cámara ─────────────────────────────────────────────────────────────────

  Future<void> _requestCameraAndStart() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _errorMessage = 'Se necesita permiso de cámara.');
      return;
    }
    if (widget.cameras.isEmpty) {
      setState(() => _errorMessage = 'No se encontraron cámaras en este dispositivo.');
      return;
    }
    await _startCamera(widget.cameras.first);
  }

  Future<void> _startCamera(CameraDescription camera) async {
    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
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

    _startAnalysisTimer();
  }

  void _startAnalysisTimer() {
    _analysisTimer?.cancel();
    _analysisTimer = Timer.periodic(_kAnalysisInterval, (_) => _analyzeFrame());
  }

  Future<void> _analyzeFrame() async {
    if (_analyzing || _controller == null || !_controller!.value.isInitialized) {
      return;
    }

    setState(() => _analyzing = true);

    try {
      final xFile = await _controller!.takePicture();
      final file = File(xFile.path);
      final result = await _service!.decide(file);

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
    } on SafeCrossException catch (e) {
      if (!mounted) return;
      setState(() => _errorMessage = e.message);
    } catch (e) {
      if (!mounted) return;
      setState(() => _errorMessage = 'Error inesperado: $e');
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
      _analysisTimer?.cancel();
      controller.dispose();
      setState(() => _cameraReady = false);
    } else if (state == AppLifecycleState.resumed) {
      _startCamera(controller.description);
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _analysisTimer?.cancel();
    _controller?.dispose();
    _tts.stop();
    _urlController.dispose();
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
          // ── Vista de cámara ─────────────────────────────────────────────
          if (_cameraReady && _controller != null)
            CameraPreview(_controller!)
          else
            _buildPlaceholder(),

          // ── Overlay de decisión (parte inferior) ───────────────────────
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: _buildDecisionPanel(),
          ),

          // ── Indicador de análisis (spinner) ────────────────────────────
          if (_analyzing)
            const Positioned(
              top: 50,
              right: 16,
              child: _AnalyzingBadge(),
            ),

          // ── Botón de ajustes ───────────────────────────────────────────
          Positioned(
            top: 44,
            left: 16,
            child: _SettingsButton(onTap: _showSettingsDialog),
          ),
        ],
      ),
    );
  }

  Widget _buildPlaceholder() {
    return Container(
      color: Colors.black,
      child: Center(
        child: _errorMessage != null
            ? Padding(
                padding: const EdgeInsets.all(32),
                child: Text(
                  _errorMessage!,
                  style: const TextStyle(color: Colors.white70, fontSize: 16),
                  textAlign: TextAlign.center,
                ),
              )
            : const CircularProgressIndicator(color: Colors.white),
      ),
    );
  }

  Widget _buildDecisionPanel() {
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
        label: 'ANALIZANDO...',
        detail: 'Apunta la cámara hacia el cruce',
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

  // ── Diálogo de ajustes ─────────────────────────────────────────────────────

  void _showSettingsDialog() {
    _urlController.text = _serverUrl;

    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Ajustes del servidor'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'URL del servidor SafeCross\n(ej: http://192.168.1.5:8000)',
              style: TextStyle(fontSize: 13),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _urlController,
              keyboardType: TextInputType.url,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                hintText: 'http://...',
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancelar'),
          ),
          FilledButton(
            onPressed: () async {
              final url = _urlController.text.trim();
              if (url.isEmpty) return;
              await _saveSettings(url);
              if (mounted) Navigator.pop(ctx);
            },
            child: const Text('Guardar'),
          ),
        ],
      ),
    );
  }
}

// ── Widgets auxiliares ────────────────────────────────────────────────────────

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
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 13,
                  ),
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
          Text('Analizando', style: TextStyle(color: Colors.white, fontSize: 12)),
        ],
      ),
    );
  }
}

class _SettingsButton extends StatelessWidget {
  final VoidCallback onTap;

  const _SettingsButton({required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(24),
        ),
        child: const Icon(Icons.settings, color: Colors.white, size: 24),
      ),
    );
  }
}
