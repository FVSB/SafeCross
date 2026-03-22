import 'package:flutter/material.dart';

/// Estado de decisión derivado de la respuesta de la API.
enum DecisionState {
  cross,        // Semáforo verde + paso peatonal → PUEDES CRUZAR
  wait,         // Semáforo rojo                  → ESPERA
  caution,      // Semáforo amarillo              → CON CUIDADO
  noCrosswalk,  // No hay paso de peatones        → SIN PASO
  noLight,      // No se detecta semáforo         → SIN SEMÁFORO
  unknown,      // Estado no determinado
}

class DecisionResult {
  final bool canCross;
  final bool crosswalkDetected;
  final int nCrosswalks;
  final bool lightDetected;
  final String? lightColor;
  final String reason;

  const DecisionResult({
    required this.canCross,
    required this.crosswalkDetected,
    required this.nCrosswalks,
    required this.lightDetected,
    required this.lightColor,
    required this.reason,
  });

  factory DecisionResult.fromJson(Map<String, dynamic> json) {
    return DecisionResult(
      canCross: json['can_cross'] as bool,
      crosswalkDetected: json['crosswalk_detected'] as bool,
      nCrosswalks: json['n_crosswalks'] as int,
      lightDetected: json['light_detected'] as bool,
      lightColor: json['light_color'] as String?,
      reason: json['reason'] as String,
    );
  }

  /// Interpreta la respuesta en uno de los estados de la UI.
  DecisionState get state {
    if (canCross) return DecisionState.cross;
    if (lightColor == 'rojo') return DecisionState.wait;
    if (lightColor == 'amarillo') return DecisionState.caution;
    if (!crosswalkDetected) return DecisionState.noCrosswalk;
    if (!lightDetected) return DecisionState.noLight;
    return DecisionState.unknown;
  }

  /// Texto principal que se muestra al usuario.
  String get mainLabel {
    switch (state) {
      case DecisionState.cross:
        return '¡PUEDES CRUZAR!';
      case DecisionState.wait:
        return 'ESPERA';
      case DecisionState.caution:
        return 'CON CUIDADO';
      case DecisionState.noCrosswalk:
        return 'SIN PASO PEATONAL';
      case DecisionState.noLight:
        return 'SEMÁFORO NO DETECTADO';
      case DecisionState.unknown:
        return 'SIN DATOS';
    }
  }

  /// Texto que lee el TTS.
  String get speechText {
    switch (state) {
      case DecisionState.cross:
        return 'Puedes cruzar. El semáforo está en verde.';
      case DecisionState.wait:
        return 'Espera. El semáforo está en rojo.';
      case DecisionState.caution:
        return 'Con cuidado. El semáforo está en amarillo.';
      case DecisionState.noCrosswalk:
        return 'No se detecta un paso de peatones. Busca un cruce seguro.';
      case DecisionState.noLight:
        return 'No se detecta semáforo. Cruza con precaución.';
      case DecisionState.unknown:
        return 'No hay suficiente información. Espera o pide ayuda.';
    }
  }

  /// Color de fondo del panel de decisión.
  Color get backgroundColor {
    switch (state) {
      case DecisionState.cross:
        return const Color(0xFF2E7D32); // verde oscuro
      case DecisionState.wait:
        return const Color(0xFFC62828); // rojo oscuro
      case DecisionState.caution:
        return const Color(0xFFF57F17); // amarillo oscuro
      case DecisionState.noCrosswalk:
        return const Color(0xFF424242); // gris oscuro
      case DecisionState.noLight:
        return const Color(0xFF37474F); // gris azulado
      case DecisionState.unknown:
        return const Color(0xFF4A148C); // morado
    }
  }

  /// Icono representativo del estado.
  IconData get icon {
    switch (state) {
      case DecisionState.cross:
        return Icons.check_circle;
      case DecisionState.wait:
        return Icons.cancel;
      case DecisionState.caution:
        return Icons.warning_rounded;
      case DecisionState.noCrosswalk:
        return Icons.remove_road;
      case DecisionState.noLight:
        return Icons.traffic;
      case DecisionState.unknown:
        return Icons.help_outline;
    }
  }
}
