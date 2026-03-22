import 'dart:io';

import 'package:http/http.dart' as http;
import 'dart:convert';

import '../models/decision_result.dart';

class SafeCrossService {
  final String baseUrl;

  SafeCrossService({required this.baseUrl});

  /// Envía [imageFile] al endpoint /decide y devuelve el resultado.
  ///
  /// Lanza [SafeCrossException] si hay error de red o respuesta inesperada.
  Future<DecisionResult> decide(File imageFile) async {
    final uri = Uri.parse('$baseUrl/decide');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(
        await http.MultipartFile.fromPath(
          'image',
          imageFile.path,
          filename: 'frame.jpg',
        ),
      );

    late http.StreamedResponse streamed;
    try {
      streamed = await request.send().timeout(const Duration(seconds: 15));
    } on SocketException {
      throw SafeCrossException(
        'No se pudo conectar al servidor. Verifica la URL en ajustes.',
      );
    } on Exception catch (e) {
      throw SafeCrossException('Error de red: $e');
    }

    final body = await streamed.stream.bytesToString();

    if (streamed.statusCode != 200) {
      final detail = _extractDetail(body);
      throw SafeCrossException('Error del servidor (${streamed.statusCode}): $detail');
    }

    try {
      final json = jsonDecode(body) as Map<String, dynamic>;
      return DecisionResult.fromJson(json);
    } catch (_) {
      throw SafeCrossException('Respuesta inesperada del servidor.');
    }
  }

  /// Comprueba que el servidor está activo (GET /health).
  Future<bool> checkHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  String _extractDetail(String body) {
    try {
      final json = jsonDecode(body) as Map<String, dynamic>;
      return json['detail']?.toString() ?? body;
    } catch (_) {
      return body;
    }
  }
}

class SafeCrossException implements Exception {
  final String message;
  SafeCrossException(this.message);

  @override
  String toString() => message;
}
