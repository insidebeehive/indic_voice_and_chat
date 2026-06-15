"""Pure-Python SIP/RTP outbound transport (for raw SIP trunks like DiDLogic).

Unlike the CPaaS adapters (Twilio/Exotel/…), a SIP trunk hands us no webhook or
media-stream WebSocket — we place the SIP INVITE and carry RTP audio ourselves.
``transport.ISipCall`` is the seam the bridge talks to (PCM16 @ 8 kHz both ways);
``pyvoip_call`` is the concrete implementation (isolated so the rest is testable).
"""
