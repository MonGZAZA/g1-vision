import sys
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

net = sys.argv[1] if len(sys.argv) > 1 else "eth0"
print(f"Testing Unitree G1 DDS on interface '{net}'...")
ChannelFactoryInitialize(0, net)

received = []

def on_state(msg: LowState_):
    received.append(time.time())
    if len(received) == 1:
        print(f"--> [SUCCESS] Received first LowState message from G1 robot!")
        try:
            print(f"    Mode: {msg.mode}, Motor states count: {len(msg.motor_state)}")
        except Exception as e:
            print(f"    Details: {e}")

sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_state, 10)

t0 = time.time()
while time.time() - t0 < 4.0:
    time.sleep(0.1)
    if len(received) >= 5:
        break

print(f"Total LowState messages received in {time.time() - t0:.1f}s: {len(received)}")
if len(received) > 0:
    print("DDS connection to G1 robot is HEALTHY and ACTIVE!")
else:
    print("No messages received on rt/lowstate yet.")
