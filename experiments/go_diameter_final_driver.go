// Independent final Diameter/S6a validation driver based on fiorix/go-diameter.
//
// This file is copied into a pinned checkout of github.com/fiorix/go-diameter
// before execution. It is intentionally separate from the Kelly/Dart
// development-domain stack used elsewhere in the repository.
package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/fiorix/go-diameter/v4/diam"
	"github.com/fiorix/go-diameter/v4/diam/avp"
	"github.com/fiorix/go-diameter/v4/diam/datatype"
	"github.com/fiorix/go-diameter/v4/diam/dict"
	"github.com/fiorix/go-diameter/v4/diam/sm"
	"github.com/fiorix/go-diameter/v4/diam/sm/smpeer"
)

const (
	vendor3GPP = uint32(10415)
	s6aAppID   = uint32(diam.TGPP_S6A_APP_ID)
	testIMSI   = "001010000000001"
	plmnID     = "\x00\xF1\x10"
	ulrFlags   = uint32(1<<1 | 1<<5)
)

var (
	mode     = flag.String("mode", "server", "server or client")
	addr     = flag.String("addr", "127.0.0.1:3869", "Diameter TCP address")
	scenario = flag.String("scenario", "benign_attach", "traffic scenario")
	count    = flag.Int("count", 2, "scenario repetitions/messages")
	delayMS  = flag.Int("delay-ms", 500, "delay between benign operations")
	timeout  = flag.Duration("timeout", 15*time.Second, "response timeout")
)

func main() {
	flag.Parse()
	rand.Seed(20261124)
	if *mode == "server" {
		if err := runServer(); err != nil {
			log.Fatal(err)
		}
		return
	}
	if *mode != "client" {
		log.Fatalf("unsupported mode %q", *mode)
	}
	if err := runClient(); err != nil {
		log.Fatal(err)
	}
}

func settings(host string) *sm.Settings {
	return &sm.Settings{
		OriginHost:       datatype.DiameterIdentity(host),
		OriginRealm:      datatype.DiameterIdentity("final.go-diameter.test"),
		VendorID:         datatype.Unsigned32(vendor3GPP),
		ProductName:      "go-diameter-final-validation",
		OriginStateID:    datatype.Unsigned32(20261124),
		FirmwareRevision: 1,
		HostIPAddresses: []datatype.Address{
			datatype.Address(net.ParseIP("127.0.0.1")),
		},
	}
}

func runServer() error {
	cfg := settings("hss.final.go-diameter.test")
	mux := sm.New(cfg)

	answer := func(c diam.Conn, m *diam.Message) {
		a := m.Answer(diam.Success)
		if sid, err := m.FindAVP(avp.SessionID, 0); err == nil && sid != nil {
			a.InsertAVP(sid)
		}
		a.NewAVP(avp.OriginHost, avp.Mbit, 0, cfg.OriginHost)
		a.NewAVP(avp.OriginRealm, avp.Mbit, 0, cfg.OriginRealm)
		a.NewAVP(avp.OriginStateID, avp.Mbit, 0, cfg.OriginStateID)
		a.NewAVP(avp.AuthSessionState, avp.Mbit, 0, datatype.Enumerated(0))
		if _, err := a.WriteTo(c); err != nil {
			log.Printf("answer write: %v", err)
		}
	}
	mux.HandleIdx(
		diam.CommandIndex{AppID: s6aAppID, Code: diam.AuthenticationInformation, Request: true},
		diam.HandlerFunc(answer))
	mux.HandleIdx(
		diam.CommandIndex{AppID: s6aAppID, Code: diam.UpdateLocation, Request: true},
		diam.HandlerFunc(answer))

	log.Printf("go-diameter final server listening on %s", *addr)
	return diam.ListenAndServeNetwork("tcp", *addr, mux, nil)
}

func runClient() error {
	cfg := settings("mme.final.go-diameter.test")
	mux := sm.New(cfg)
	done := make(chan uint32, 8192)

	answerHandler := func(c diam.Conn, m *diam.Message) {
		var rc uint32
		if rcAVP, err := m.FindAVP(avp.ResultCode, 0); err == nil && rcAVP != nil {
			if u, ok := rcAVP.Data.(datatype.Unsigned32); ok {
				rc = uint32(u)
			}
		}
		done <- rc
	}
	mux.HandleIdx(
		diam.CommandIndex{AppID: s6aAppID, Code: diam.AuthenticationInformation, Request: false},
		diam.HandlerFunc(answerHandler))
	mux.HandleIdx(
		diam.CommandIndex{AppID: s6aAppID, Code: diam.UpdateLocation, Request: false},
		diam.HandlerFunc(answerHandler))

	cli := &sm.Client{
		Dict:               dict.Default,
		Handler:            mux,
		MaxRetransmits:     2,
		RetransmitInterval: time.Second,
		EnableWatchdog:     false,
		SupportedVendorID: []*diam.AVP{
			diam.NewAVP(avp.SupportedVendorID, avp.Mbit, 0,
				datatype.Unsigned32(vendor3GPP)),
		},
		VendorSpecificApplicationID: []*diam.AVP{
			diam.NewAVP(avp.VendorSpecificApplicationID, avp.Mbit, 0, &diam.GroupedAVP{
				AVP: []*diam.AVP{
					diam.NewAVP(avp.AuthApplicationID, avp.Mbit, 0,
						datatype.Unsigned32(s6aAppID)),
					diam.NewAVP(avp.VendorID, avp.Mbit, 0,
						datatype.Unsigned32(vendor3GPP)),
				},
			}),
		},
	}

	conn, err := cli.DialNetwork("tcp", *addr)
	if err != nil {
		return err
	}
	defer conn.Close()

	meta, ok := smpeer.FromContext(conn.Context())
	if !ok {
		return errors.New("peer metadata unavailable")
	}

	ops, paced, err := scenarioOps(strings.ToLower(*scenario), *count)
	if err != nil {
		return err
	}
	delay := time.Duration(*delayMS) * time.Millisecond

	if paced {
		for _, op := range ops {
			if err := sendOp(conn, cfg, meta, op); err != nil {
				return err
			}
			if err := waitSuccess(done); err != nil {
				return err
			}
			if delay > 0 {
				time.Sleep(delay)
			}
		}
	} else {
		for _, op := range ops {
			if err := sendOp(conn, cfg, meta, op); err != nil {
				return err
			}
		}
		for range ops {
			if err := waitSuccess(done); err != nil {
				return err
			}
		}
	}
	time.Sleep(150 * time.Millisecond)
	return nil
}

func scenarioOps(name string, n int) ([]string, bool, error) {
	if n < 1 {
		n = 1
	}
	ops := make([]string, 0, n*2)
	switch name {
	case "benign_attach":
		for i := 0; i < n; i++ {
			ops = append(ops, "air", "ulr")
		}
		return ops, true, nil
	case "benign_reauth":
		for i := 0; i < n; i++ {
			ops = append(ops, "air")
		}
		return ops, true, nil
	case "benign_update":
		for i := 0; i < n; i++ {
			ops = append(ops, "ulr")
		}
		return ops, true, nil
	case "air_flood":
		for i := 0; i < n; i++ {
			ops = append(ops, "air")
		}
		return ops, false, nil
	case "ulr_flood":
		for i := 0; i < n; i++ {
			ops = append(ops, "ulr")
		}
		return ops, false, nil
	case "mixed_flood":
		for i := 0; i < n; i++ {
			if i%2 == 0 {
				ops = append(ops, "air")
			} else {
				ops = append(ops, "ulr")
			}
		}
		return ops, false, nil
	case "tracking_burst":
		for i := 0; i < n; i++ {
			switch i % 3 {
			case 0, 1:
				ops = append(ops, "air")
			default:
				ops = append(ops, "ulr")
			}
		}
		return ops, false, nil
	default:
		return nil, false, fmt.Errorf("unknown scenario %q", name)
	}
}

func waitSuccess(done <-chan uint32) error {
	select {
	case rc := <-done:
		if rc != diam.Success {
			return fmt.Errorf("non-success Result-Code=%d", rc)
		}
		return nil
	case <-time.After(*timeout):
		return errors.New("Diameter response timeout")
	}
}

func sendOp(c diam.Conn, cfg *sm.Settings, meta *smpeer.Metadata, op string) error {
	switch op {
	case "air":
		return sendAIR(c, cfg, meta)
	case "ulr":
		return sendULR(c, cfg, meta)
	default:
		return fmt.Errorf("unknown operation %q", op)
	}
}

func sid() datatype.UTF8String {
	return datatype.UTF8String(
		"final-session;" + strconv.FormatUint(rand.Uint64(), 10))
}

func commonAVPs(m *diam.Message, cfg *sm.Settings, meta *smpeer.Metadata) {
	m.NewAVP(avp.SessionID, avp.Mbit, 0, sid())
	m.NewAVP(avp.OriginHost, avp.Mbit, 0, cfg.OriginHost)
	m.NewAVP(avp.OriginRealm, avp.Mbit, 0, cfg.OriginRealm)
	m.NewAVP(avp.DestinationRealm, avp.Mbit, 0, meta.OriginRealm)
	m.NewAVP(avp.DestinationHost, avp.Mbit, 0, meta.OriginHost)
	m.NewAVP(avp.UserName, avp.Mbit, 0, datatype.UTF8String(testIMSI))
	m.NewAVP(avp.AuthSessionState, avp.Mbit, 0, datatype.Enumerated(0))
}

func sendAIR(c diam.Conn, cfg *sm.Settings, meta *smpeer.Metadata) error {
	m := diam.NewRequest(
		diam.AuthenticationInformation, s6aAppID, dict.Default)
	commonAVPs(m, cfg, meta)
	m.NewAVP(avp.VisitedPLMNID, avp.Vbit|avp.Mbit, vendor3GPP,
		datatype.OctetString(plmnID))
	m.NewAVP(avp.RequestedEUTRANAuthenticationInfo,
		avp.Vbit|avp.Mbit, vendor3GPP, &diam.GroupedAVP{
			AVP: []*diam.AVP{
				diam.NewAVP(avp.NumberOfRequestedVectors,
					avp.Vbit|avp.Mbit, vendor3GPP,
					datatype.Unsigned32(1)),
				diam.NewAVP(avp.ImmediateResponsePreferred,
					avp.Vbit|avp.Mbit, vendor3GPP,
					datatype.Unsigned32(0)),
			},
		})
	_, err := m.WriteTo(c)
	return err
}

func sendULR(c diam.Conn, cfg *sm.Settings, meta *smpeer.Metadata) error {
	m := diam.NewRequest(diam.UpdateLocation, s6aAppID, dict.Default)
	commonAVPs(m, cfg, meta)
	m.NewAVP(avp.RATType, avp.Mbit, vendor3GPP,
		datatype.Enumerated(1004))
	m.NewAVP(avp.ULRFlags, avp.Vbit|avp.Mbit, vendor3GPP,
		datatype.Unsigned32(ulrFlags))
	m.NewAVP(avp.VisitedPLMNID, avp.Vbit|avp.Mbit, vendor3GPP,
		datatype.OctetString(plmnID))
	_, err := m.WriteTo(c)
	return err
}

func init() {
	log.SetOutput(os.Stderr)
}
