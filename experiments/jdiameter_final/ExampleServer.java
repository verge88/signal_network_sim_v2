package org.example.server;

import java.io.InputStream;
import org.jdiameter.api.Answer;
import org.jdiameter.api.ApplicationId;
import org.jdiameter.api.Configuration;
import org.jdiameter.api.Network;
import org.jdiameter.api.NetworkReqListener;
import org.jdiameter.api.Request;
import org.jdiameter.api.SessionFactory;
import org.jdiameter.api.Stack;
import org.jdiameter.server.impl.StackImpl;
import org.jdiameter.server.impl.helpers.XMLConfiguration;

/**
 * Frozen final-confirmation S6a responder using the RestComm jDiameter stack.
 * It intentionally performs only protocol-level request/answer handling; the
 * detector is evaluated from the actual TCP wire capture, not application logs.
 */
public final class ExampleServer implements NetworkReqListener {
  private static final long S6A_APP_ID = 16777251L;
  private Stack stack;

  private void start() throws Exception {
    stack = new StackImpl();
    InputStream in = ExampleServer.class.getClassLoader().getResourceAsStream(
        "org/example/server/server-jdiameter-config.xml");
    if (in == null) {
      throw new IllegalStateException("server config missing");
    }
    Configuration config = new XMLConfiguration(in);
    SessionFactory factory = stack.init(config);
    if (factory == null) {
      throw new IllegalStateException("jDiameter session factory unavailable");
    }
    Network network = stack.unwrap(Network.class);
    network.addNetworkReqListener(
        this, ApplicationId.createByAuthAppId(S6A_APP_ID));
    stack.start();
    System.out.println("JDIAMETER_FINAL_SERVER_READY");
  }

  @Override
  public Answer processRequest(Request request) {
    int code = request.getCommandCode();
    if (code != 316 && code != 318 && code != 319) {
      System.err.println("Unexpected S6a command: " + code);
    }
    return request.createAnswer(2001);
  }

  public static void main(String[] args) throws Exception {
    ExampleServer server = new ExampleServer();
    server.start();
    while (true) {
      Thread.sleep(1000L);
    }
  }
}
