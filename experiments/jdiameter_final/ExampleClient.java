package org.example.client;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.jdiameter.api.Answer;
import org.jdiameter.api.ApplicationId;
import org.jdiameter.api.Configuration;
import org.jdiameter.api.EventListener;
import org.jdiameter.api.Network;
import org.jdiameter.api.NetworkReqListener;
import org.jdiameter.api.Request;
import org.jdiameter.api.Session;
import org.jdiameter.api.SessionFactory;
import org.jdiameter.api.Stack;
import org.jdiameter.server.impl.StackImpl;
import org.jdiameter.server.impl.helpers.XMLConfiguration;

/**
 * Frozen final-confirmation jDiameter S6a client.
 *
 * Scenarios intentionally use only protocol-valid S6a command codes. The
 * detector never receives scenario metadata; it sees only the captured wire.
 */
public final class ExampleClient implements EventListener<Request, Answer> {
  private static final long S6A_APP_ID = 16777251L;
  private static final String REALM = "exchange.example.org";
  private static final String SERVER_URI = "aaa://127.0.0.1:3868";

  private Stack stack;
  private SessionFactory factory;
  private Session session;
  private CountDownLatch latch;
  private final AtomicInteger failures = new AtomicInteger();

  private void startStack() throws Exception {
    stack = new StackImpl();
    InputStream in = ExampleClient.class.getClassLoader().getResourceAsStream(
        "org/example/client/client-jdiameter-config.xml");
    if (in == null) {
      throw new IllegalStateException("client config missing");
    }
    Configuration config = new XMLConfiguration(in);
    factory = stack.init(config);
    Network network = stack.unwrap(Network.class);
    network.addNetworkReqListener(new NetworkReqListener() {
      @Override
      public Answer processRequest(Request request) {
        return request.createAnswer(2001);
      }
    }, ApplicationId.createByAuthAppId(S6A_APP_ID));
    stack.start();
    Thread.sleep(1800L);
  }

  private static List<Integer> commands(String scenario, int count) {
    List<Integer> out = new ArrayList<Integer>();
    for (int i = 0; i < count; i++) {
      if ("benign_attach".equals(scenario)) {
        out.add(318);
        out.add(316);
      } else if ("benign_reauth".equals(scenario)) {
        out.add(318);
      } else if ("benign_update".equals(scenario)) {
        out.add(316);
      } else if ("air_flood".equals(scenario)) {
        out.add(318);
      } else if ("ulr_flood".equals(scenario)) {
        out.add(316);
      } else if ("mixed_flood".equals(scenario)) {
        out.add((i % 2 == 0) ? 318 : 316);
      } else if ("tracking_burst".equals(scenario)) {
        int m = i % 5;
        out.add(m < 2 ? 318 : (m < 4 ? 316 : 319));
      } else {
        throw new IllegalArgumentException("unknown scenario: " + scenario);
      }
    }
    return out;
  }

  private void runScenario(String scenario, int count, long delayMs)
      throws Exception {
    List<Integer> codes = commands(scenario, count);
    latch = new CountDownLatch(codes.size());
    session = factory.getNewSession(
        "jdiameter-final;" + scenario + ";" + System.nanoTime());
    ApplicationId app = ApplicationId.createByAuthAppId(S6A_APP_ID);

    for (Integer code : codes) {
      Request request = session.createRequest(code, app, REALM, SERVER_URI);
      session.send(request, this);
      if (delayMs > 0) {
        Thread.sleep(delayMs);
      }
    }

    if (!latch.await(Math.max(15L, codes.size() / 2L), TimeUnit.SECONDS)) {
      throw new IllegalStateException(
          "Timed out waiting for Diameter answers; remaining=" + latch.getCount());
    }
    if (failures.get() != 0) {
      throw new IllegalStateException(
          "Diameter request failures=" + failures.get());
    }
    session.release();
  }

  @Override
  public void receivedSuccessMessage(Request request, Answer answer) {
    if (answer == null || answer.getCommandCode() != request.getCommandCode()) {
      failures.incrementAndGet();
    }
    latch.countDown();
  }

  @Override
  public void timeoutExpired(Request request) {
    failures.incrementAndGet();
    latch.countDown();
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 3) {
      System.err.println("usage: scenario count delay_ms");
      System.exit(2);
    }
    String scenario = args[0];
    int count = Integer.parseInt(args[1]);
    long delayMs = Long.parseLong(args[2]);

    ExampleClient client = new ExampleClient();
    client.startStack();
    client.runScenario(scenario, count, delayMs);
    System.out.println("JDIAMETER_FINAL_CLIENT_OK " + scenario + " " + count);
    System.exit(0);
  }
}
