using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;

//指令控制
public class Command_control
{
    private Light_control LightControl = new Light_control();
    private VoiceOutputForTesting VoiceTest = new VoiceOutputForTesting();
    public bool choose_command(String index){
        bool successful = false;
        switch(index){
            case "1":
                LightControl.Switch();
                //VoiceTest.test1();
                successful = true;
                break;
            case "2":
                LightControl.Switch();
                successful = true;
                break;
            default:
                //VoiceTest.test3();
                successful = false;
                break;
        }
        return successful;
   }
}


