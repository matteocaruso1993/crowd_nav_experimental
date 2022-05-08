from pynput import keyboard
import rospy


class KeyListener:
    def __init__(self):
        self.keyboard = None
        self.controller = keyboard.Controller()
        self._skip_input = False
        
    def _on_press(self,key):
        print('{0}'.format(
        key))
        pass
    def _on_release(self,key):
        pass
    
    def initialize(self):
        self.keyboard = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
    
    
    def start(self):
        self.keyboard.start()
    
    def stop(self):
        self.keyboard.stop()
        
        
class RobotKeyListener(KeyListener):
    def __init__(self):
        super().__init__()
        self.stop_triggered = True
        self.emergency_stop_triggered = False
        self.new_target = None
        self.homing_requested = False
        self.path_requested = False
        self.new_path_enabled = False
        self.catched = True
        
    def _on_press(self,key):
        if not self._skip_input:
            if key==keyboard.Key.space:
                rospy.logwarn('Temporary stop Triggered')
                self.stop_triggered = not self.stop_triggered

            
            if key == keyboard.Key.enter and self.emergency_stop_triggered:
                self.emergency_stop_triggered = False

            if key == keyboard.Key.enter  and self.path_requested:
                self.path_requested = False
                rospy.loginfo('Registered Points')


            try:
                if key.char == 's':
                    self.emergency_stop_triggered = True
                    rospy.logwarn('Emergency stop has been triggered! Press ENTER in order to unlock the emergency stop.')
                
                if key.char == 'n':
                    self._new_target_requested()

                if key.char == 'h':
                    self.homing_requested = True

                if key.char == 'p':
                    if self.new_path_enabled:
                        self.catched = False
                        self.path_requested = True
                        rospy.loginfo('Listening to RVIZ point input. Press ENTER to stop the listener')
                    

            except:
                pass
        
    
        
    
    def _new_target_requested(self):
        #self.controller.press(keyboard.Key.space)
        #self.controller.release(keyboard.Key.space)
        self.stop_triggered = True
        self._skip_input = True
        while True:
            print('\n')
            x, y = input('Enter new x target:'), input('Enter new y target:')
            if x[0] == 'n':
                x = x[1:]
            try:
                x = float(x)
                y = float(y)
                self.new_target = [x,y]
                self._skip_input = False
                break
            except:
                rospy.logerr('Invalid input data. Insert again...')
                



    

if __name__ == '__main__':
    try:
        global key_listened
        #k = KeyListener()
        k = RobotKeyListener() 
        k.initialize()
        k.start()
        while True:
            continue
    except KeyboardInterrupt:
        print('Quitting')
        k.stop()
    except:
        print('Quitting 2')
    
    
    
    
    
        

