# pepper_client/bridge_server.py
# ============================================
# Pepper Robot Bridge Server (Python 2.7)
# ============================================
# This code runs ON the Pepper robot itself.
# It provides a REST API that the NeuroPepper
# server can call to control the robot.
#
# IMPORTANT: This uses Python 2.7 and NAOqi SDK
#
# CHANGES FROM ORIGINAL:
# - ROBOT_IP updated to real Pepper IP: 10.186.13.46
# - Flask default port changed from 9559 to 5009
#   (9559 is reserved for NAOqi, cannot be reused)
# ============================================

from __future__ import print_function
import argparse
import json
import sys

# Flask 1.x works with Python 2.7
try:
    from flask import Flask, request, jsonify
except ImportError:
    print("Flask not found. Install with: pip install flask==1.1.4")
    sys.exit(1)

# NAOqi SDK
try:
    from naoqi import ALProxy
except ImportError:
    print("NAOqi SDK not found. Running in simulation mode.")
    ALProxy = None

app = Flask(__name__)

# ============================================
# ROBOT IP AND PORT CONFIGURATION
# ROBOT_IP  = your Pepper's real IP address
# ROBOT_PORT = NAOqi port, always 9559, never change this
# ============================================
ROBOT_IP = "10.186.13.46"
ROBOT_PORT = 9559

# Proxy cache
_proxies = {}


def get_proxy(name):
    """Get or create an ALProxy for a service."""
    if ALProxy is None:
        return None

    if name not in _proxies:
        try:
            _proxies[name] = ALProxy(name, ROBOT_IP, ROBOT_PORT)
        except Exception as e:
            print("Error creating proxy {}: {}".format(name, e))
            return None

    return _proxies[name]


# ============================================
# STATUS ENDPOINT
# ============================================

@app.route('/status', methods=['GET'])
def get_status():
    """Get robot status."""
    battery = None

    if ALProxy:
        try:
            battery_proxy = get_proxy("ALBattery")
            if battery_proxy:
                battery = battery_proxy.getBatteryCharge()
        except:
            pass

    return jsonify({
        "connected": ALProxy is not None,
        "battery": battery,
        "simulation": ALProxy is None
    })


# ============================================
# SPEECH ENDPOINTS
# ============================================

@app.route('/speak', methods=['POST'])
def speak():
    """Make the robot speak."""
    data = request.get_json() or {}
    text = data.get('text', '')
    speed = data.get('speed', 100)
    pitch = data.get('pitch', 100)

    if not text:
        return jsonify({"success": False, "error": "No text provided"})

    if ALProxy:
        try:
            # Use animated speech for gestures
            tts = get_proxy("ALAnimatedSpeech")
            if tts:
                tts.say(str(text))
                return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    # Simulation mode
    print("[SPEAK] {}".format(text))
    return jsonify({"success": True, "simulation": True})


@app.route('/animated_speech', methods=['POST'])
def animated_speech():
    """Speak with automatic gestures."""
    data = request.get_json() or {}
    text = data.get('text', '')
    mode = data.get('mode', 'contextual')

    if ALProxy:
        try:
            tts = get_proxy("ALAnimatedSpeech")
            if tts:
                config = {"bodyLanguageMode": mode}
                tts.say(str(text), config)
                return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    print("[ANIMATED_SPEECH] {}".format(text))
    return jsonify({"success": True, "simulation": True})


# ============================================
# MOVEMENT ENDPOINTS
# ============================================

@app.route('/move', methods=['POST'])
def move():
    """Move the robot."""
    data = request.get_json() or {}
    x = float(data.get('x', 0))
    y = float(data.get('y', 0))
    theta = float(data.get('theta', 0))

    if ALProxy:
        try:
            motion = get_proxy("ALMotion")
            if motion:
                motion.moveTo(x, y, theta)
                return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    print("[MOVE] x={}, y={}, theta={}".format(x, y, theta))
    return jsonify({"success": True, "simulation": True})


@app.route('/move_to', methods=['POST'])
def move_to():
    """Move to a position."""
    return move()


@app.route('/stop', methods=['POST'])
def stop():
    """Stop all movement."""
    if ALProxy:
        try:
            motion = get_proxy("ALMotion")
            if motion:
                motion.stopMove()
                return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    print("[STOP]")
    return jsonify({"success": True, "simulation": True})


# ============================================
# GESTURE ENDPOINTS
# ============================================

@app.route('/gesture', methods=['POST'])
def gesture():
    """Play a gesture animation."""
    data = request.get_json() or {}
    animation = data.get('animation', '')

    if not animation:
        return jsonify({"success": False, "error": "No animation specified"})

    if ALProxy:
        try:
            behavior = get_proxy("ALBehaviorManager")
            if behavior:
                if behavior.isBehaviorInstalled(animation):
                    behavior.runBehavior(animation)
                else:
                    # Try as animation
                    anim = get_proxy("ALAnimationPlayer")
                    if anim:
                        anim.run(animation)
                return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    print("[GESTURE] {}".format(animation))
    return jsonify({"success": True, "simulation": True})


# ============================================
# LED ENDPOINTS
# ============================================

@app.route('/leds', methods=['POST'])
def leds():
    """Control LED colors."""
    data = request.get_json() or {}
    group = data.get('group', 'FaceLeds')
    color = data.get('color', 'white')

    # Map friendly names to LED groups
    group_map = {
        'eyes': 'FaceLeds',
        'ears': 'EarLeds',
        'chest': 'ChestLeds',
        'all': 'AllLeds'
    }
    led_group = group_map.get(group.lower(), group)

    # Map color names to RGB values (0-1)
    color_map = {
        'white': (1, 1, 1),
        'red': (1, 0, 0),
        'green': (0, 1, 0),
        'blue': (0, 0, 1),
        'yellow': (1, 1, 0),
        'cyan': (0, 1, 1),
        'magenta': (1, 0, 1),
        'off': (0, 0, 0)
    }
    rgb = color_map.get(color.lower(), (1, 1, 1))

    if ALProxy:
        try:
            leds_proxy = get_proxy("ALLeds")
            if leds_proxy:
                # Convert RGB to hex for some LED methods
                hex_color = int('%02x%02x%02x' % (
                    int(rgb[0]*255),
                    int(rgb[1]*255),
                    int(rgb[2]*255)
                ), 16)
                leds_proxy.fadeRGB(led_group, hex_color, 0.5)
                return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    print("[LEDS] {} = {}".format(led_group, color))
    return jsonify({"success": True, "simulation": True})


# ============================================
# TABLET ENDPOINTS
# ============================================

@app.route('/tablet', methods=['POST'])
def tablet():
    """Display content on tablet."""
    data = request.get_json() or {}
    content = data.get('content', '')
    content_type = data.get('type', 'text')
    action = data.get('action')

    if action == 'hide':
        if ALProxy:
            try:
                tablet = get_proxy("ALTabletService")
                if tablet:
                    tablet.hideWebview()
                    return jsonify({"success": True})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        print("[TABLET] Hide")
        return jsonify({"success": True, "simulation": True})

    if ALProxy:
        try:
            tablet = get_proxy("ALTabletService")
            if tablet:
                if content_type == 'web':
                    tablet.showWebview(content)
                elif content_type == 'image':
                    tablet.showImage(content)
                else:
                    # Show text as HTML
                    html = "<html><body style='font-size:24px;padding:20px;'>{}</body></html>".format(
                        content)
                    tablet.showWebview("data:text/html," + html)
                return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    print("[TABLET] {} ({})".format(content[:50], content_type))
    return jsonify({"success": True, "simulation": True})


# ============================================
# TRACKING ENDPOINTS
# ============================================

@app.route('/track', methods=['POST'])
def track():
    """Start/stop tracking."""
    data = request.get_json() or {}
    target = data.get('target', 'face')

    if ALProxy:
        try:
            if target == 'stop':
                tracker = get_proxy("ALTracker")
                if tracker:
                    tracker.stopTracker()
            elif target == 'face':
                tracker = get_proxy("ALTracker")
                if tracker:
                    tracker.registerTarget("Face", 0.2)
                    tracker.track("Face")
            elif target == 'sound':
                tracker = get_proxy("ALSoundLocalization")
                # Configure sound tracking
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    print("[TRACK] {}".format(target))
    return jsonify({"success": True, "simulation": True})


# ============================================
# AUTONOMOUS LIFE CONTROL
# ============================================

@app.route('/autonomous_life', methods=['POST'])
def autonomous_life():
    """Control autonomous life."""
    data = request.get_json() or {}
    state = data.get('state', 'disabled')

    if ALProxy:
        try:
            life = get_proxy("ALAutonomousLife")
            if life:
                life.setState(state)
                return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    print("[AUTONOMOUS_LIFE] {}".format(state))
    return jsonify({"success": True, "simulation": True})


@app.route('/breathing', methods=['POST'])
def breathing():
    """Enable/disable breathing animation."""
    data = request.get_json() or {}
    enabled = data.get('enabled', True)

    if ALProxy:
        try:
            motion = get_proxy("ALMotion")
            if motion:
                motion.setBreathEnabled("Body", enabled)
                return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    print("[BREATHING] {}".format("enabled" if enabled else "disabled"))
    return jsonify({"success": True, "simulation": True})


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Pepper Robot Bridge Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    # PORT EXPLANATION:
    # 5009 = the door number for THIS Flask bridge server
    # 9559 = NAOqi's door (already taken by the robot, never use this for Flask)
    parser.add_argument("--port", type=int, default=5009,
                        help="Port to run on (default: 5009)")
    args = parser.parse_args()

    print("=" * 50)
    print("Pepper Robot Bridge Server")
    print("=" * 50)
    print("Host: {}".format(args.host))
    print("Flask Bridge Port: {}".format(args.port))
    print("Pepper IP: {}".format(ROBOT_IP))
    print("NAOqi Port: {}".format(ROBOT_PORT))
    print("NAOqi: {}".format("Available" if ALProxy else "Not available (simulation)"))
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
