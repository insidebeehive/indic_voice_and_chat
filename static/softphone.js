/**
 * softphone.js — one browser API over Twilio Voice and Stringee Web SDKs.
 *
 * The CRM's *backend* mints a token (POST /api/v1/softphone/token, with the
 * tenant bearer) and hands the JSON to the browser. This helper takes that JSON
 * and gives you one provider-agnostic API:
 *
 *     const phone = await Softphone.create(tokenResponse);   // the endpoint's JSON
 *     phone.on('status', s => console.log(s));                // ringing|connected|disconnected
 *     await phone.dial('+9118XXXXXXXX');                      // the lead number
 *     phone.hangup();
 *
 * The provider branch lives here, not in the CRM's app. `tokenResponse` is the
 * exact object returned by /api/v1/softphone/token:
 *     { provider: "twilio"|"stringee", token, identity, ttl_seconds, params }
 *
 * SDKs: this expects the provider's SDK to be available — either already on the
 * page, or auto-loaded by passing `opts.twilioSdkUrl` / `opts.stringeeSdkUrl`.
 *   - Twilio Voice JS SDK (global `Twilio.Device`, or pass a URL/bundle)
 *   - Stringee Web SDK (globals `StringeeClient` + `StringeeCall`)
 *
 * Normalized status vocabulary (the `status` event payload):
 *   "registering" | "ready" | "calling" | "ringing" | "connected" |
 *   "disconnected" | "error"
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Softphone = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // The global object, captured inside the factory (the outer IIFE's `root`
  // param is not in this scope). Provider SDKs register their globals here.
  const glob = (typeof window !== "undefined") ? window
    : (typeof self !== "undefined") ? self
    : (typeof globalThis !== "undefined") ? globalThis : this;

  // Default vendor SDK CDNs, used when the page neither pre-loads the SDK nor
  // passes a *SdkUrl. Override via Softphone.create(tok, { stringeeSdkUrl, twilioSdkUrl })
  // to pin a specific version.
  const DEFAULT_SDK = {
    stringee: "https://cdn.stringee.com/sdk/web/latest/stringee-web-sdk.min.js",
    twilio: "https://sdk.twilio.com/js/voice/releases/2.11.2/twilio.min.js",
  };

  function loadScript(url) {
    return new Promise(function (resolve, reject) {
      if (!url) return reject(new Error("no SDK url provided and SDK global not found"));
      const s = document.createElement("script");
      s.src = url;
      s.async = true;
      s.onload = resolve;
      s.onerror = function () { reject(new Error("failed to load " + url)); };
      document.head.appendChild(s);
    });
  }

  // Tiny event emitter so callers can phone.on('status'|'error'|'incoming', fn).
  function Emitter() { this._h = {}; }
  Emitter.prototype.on = function (evt, fn) {
    (this._h[evt] = this._h[evt] || []).push(fn);
    return this;
  };
  Emitter.prototype.emit = function (evt, payload) {
    (this._h[evt] || []).forEach(function (fn) {
      try { fn(payload); } catch (e) { /* a listener error must not break the call */ }
    });
  };

  function Softphone(provider, impl, emitter) {
    this.provider = provider;
    this._impl = impl;       // { dial, hangup, destroy }
    this._em = emitter;
  }
  Softphone.prototype.on = function (evt, fn) { this._em.on(evt, fn); return this; };
  Softphone.prototype.dial = function (toNumber, params) { return this._impl.dial(toNumber, params || {}); };
  Softphone.prototype.hangup = function () { return this._impl.hangup(); };
  Softphone.prototype.destroy = function () { return this._impl.destroy(); };

  /**
   * Build a Softphone from the /softphone/token response.
   * @param {object} tokenResponse  { provider, token, identity, ... }
   * @param {object} [opts]         { twilioSdkUrl, stringeeSdkUrl, edge, debug }
   */
  Softphone.create = async function (tokenResponse, opts) {
    opts = opts || {};
    if (!tokenResponse || !tokenResponse.provider || !tokenResponse.token) {
      throw new Error("Softphone.create needs the /softphone/token response { provider, token, ... }");
    }
    const em = new Emitter();
    const provider = String(tokenResponse.provider).toLowerCase();
    let impl;
    if (provider === "twilio") impl = await buildTwilio(tokenResponse, opts, em);
    else if (provider === "stringee") impl = await buildStringee(tokenResponse, opts, em);
    else throw new Error("unsupported softphone provider: " + provider);
    return new Softphone(provider, impl, em);
  };

  // --- Twilio ---------------------------------------------------------------
  async function buildTwilio(tok, opts, em) {
    let Device = (glob.Twilio && glob.Twilio.Device) || (glob.Device);
    if (!Device) {
      await loadScript(opts.twilioSdkUrl || DEFAULT_SDK.twilio);
      Device = (glob.Twilio && glob.Twilio.Device) || glob.Device;
    }
    if (!Device) throw new Error("Twilio Voice SDK not found (set opts.twilioSdkUrl)");

    em.emit("status", "registering");
    const device = new Device(tok.token, { edge: opts.edge, logLevel: opts.debug ? 1 : undefined });
    device.on("registered", function () { em.emit("status", "ready"); });
    device.on("error", function (e) { em.emit("error", e); em.emit("status", "error"); });
    device.on("incoming", function (call) { em.emit("incoming", call); });
    // refresh the short-lived token in place if the page provides a minter
    device.on("tokenWillExpire", async function () {
      if (typeof opts.onTokenExpiring === "function") {
        try { device.updateToken(await opts.onTokenExpiring()); } catch (e) { em.emit("error", e); }
      }
    });
    await device.register();

    let current = null;
    return {
      dial: async function (to, params) {
        current = await device.connect({ params: Object.assign({ To: to }, params) });
        current.on("ringing", function () { em.emit("status", "ringing"); });
        current.on("accept", function () { em.emit("status", "connected"); });
        current.on("disconnect", function () { em.emit("status", "disconnected"); current = null; });
        current.on("cancel", function () { em.emit("status", "disconnected"); current = null; });
        current.on("error", function (e) { em.emit("error", e); });
        em.emit("status", "calling");
        return current;
      },
      hangup: function () { if (current) current.disconnect(); else device.disconnectAll(); },
      destroy: function () { try { device.destroy(); } catch (e) {} },
    };
  }

  // --- Stringee -------------------------------------------------------------
  // Stringee SignalingState: CALLING=0, RINGING=1, ANSWERED=2, BUSY=3, ENDED=4.
  const STRINGEE_SIGNAL = {
    0: "calling", 1: "ringing", 2: "connected", 3: "disconnected", 4: "disconnected",
  };

  async function buildStringee(tok, opts, em) {
    if (!glob.StringeeClient || !glob.StringeeCall) {
      await loadScript(opts.stringeeSdkUrl || DEFAULT_SDK.stringee);
    }
    const StringeeClient = glob.StringeeClient, StringeeCall = glob.StringeeCall;
    if (!StringeeClient || !StringeeCall) {
      throw new Error("Stringee Web SDK not found (set opts.stringeeSdkUrl)");
    }

    em.emit("status", "registering");
    const client = new StringeeClient();
    const identity = tok.identity;
    await new Promise(function (resolve, reject) {
      client.on("authen", function (res) {
        // res.r === 0 → authenticated
        if (res && res.r === 0) { em.emit("status", "ready"); resolve(); }
        else { em.emit("status", "error"); reject(new Error("stringee auth failed: " + JSON.stringify(res))); }
      });
      client.on("disconnect", function () { em.emit("status", "disconnected"); });
      client.connect(tok.token);
    });

    let current = null;
    function wire(call) {
      call.on("signalingstate", function (state) {
        const code = state && (state.code != null ? state.code : state);
        const mapped = STRINGEE_SIGNAL[code];
        if (mapped) em.emit("status", mapped);
        if (mapped === "disconnected") current = null;
      });
      call.on("error", function (e) { em.emit("error", e); });
    }
    return {
      dial: function (to, params) {
        // App-to-phone: from = the agent identity, to = the lead number.
        current = new StringeeCall(client, identity, to);
        current.isPhoneCall = true;
        // Stringee does NOT forward `to` to our Answer URL for app-to-phone calls
        // (it arrives empty, fromInternal=true) — so carry the destination in
        // customData, which Stringee delivers to the Answer URL as `custom`.
        current.customDataFromYourServer =
          (params && params.customData) || JSON.stringify({ to: to });
        wire(current);
        em.emit("status", "calling");
        return new Promise(function (resolve, reject) {
          current.makeCall(function (res) {
            if (res && res.r === 0) resolve(current);
            else { em.emit("status", "error"); reject(new Error("stringee makeCall failed: " + JSON.stringify(res))); }
          });
        });
      },
      hangup: function () { if (current) current.hangup(function () {}); },
      destroy: function () { try { client.disconnect(); } catch (e) {} },
    };
  }

  return Softphone;
});
