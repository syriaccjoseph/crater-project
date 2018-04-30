import crater_loader as cl
import crater_network as cn
import signal
import sys

td, ted = cl.load_data()
net = cn.Network([40000, 10, 4, 1])

def sigterm_handler(signal, frame):
    # save the state here or do whatever you want
    print 'False pos ', len(net.false_pos)
    print 'False neg : ', len(net.false_neg)
    print 'False pos'
    for img in net.false_pos :
        print img
    print 'False neg'
    for img in net.false_neg :
        print img
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

try :
    net.SGD(td, 60, 5, 0.2, ted)

except KeyboardInterrupt :
    print 'Keyboard Interrupt'

finally:
    print 'False pos ', len(net.false_pos)
    print 'False neg : ', len(net.false_neg)
    print 'False pos'
    for img in net.false_pos :
        print img
    print 'False neg'
    for img in net.false_neg :
        print img