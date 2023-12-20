using Microsoft.CognitiveServices.Speech.Audio;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using Microsoft.CognitiveServices.Speech;
using Newtonsoft.Json.Linq;
using static ChatGPT;
using System.IO;
//using TreeEditor;
using System.Linq;
using System.Drawing;
public class VoiceOutputForTesting
{
     //AI語音播放器
    private text_to_voice Speaker;
    //語音辨識工具
    private Microsoft.CognitiveServices.Speech.SpeechConfig configuration;
    //SpeechRecognizer 物件
    private SpeechRecognizer recognizer;
    [SerializeField] private List<ApiKeyData> ApiKey = new List<ApiKeyData>();
    void Start(){
        //讀取鑰匙
        try
        {
            var inputString = File.ReadAllText("Keys.json");
            ApiKey = JsonUtility.FromJson<Serialization<ApiKeyData>>(inputString).ToList();
        }
        //無鑰匙 回報
        catch (Exception e)
        {
            Debug.Log("No have Key file.");
        }

        //AI語音播放器
        Speaker = new text_to_voice(ApiKey[0].key, ApiKey[0].region);
        configuration = SpeechConfig.FromSubscription(ApiKey[1].key, ApiKey[1].region);
        configuration.SpeechRecognitionLanguage = "zh-TW";
    }
    public void test1(){
        string callback = "這是一號指令";
        //Speaker.speak(callback);
        Debug.Log(callback);
    }
    public void test2(){
        string callback = "這是二號指令";
        //Speaker.speak(callback);
        Debug.Log(callback);
    }
    public void test3(){
        string callback = "很抱歉，我沒有此服務";
        //Speaker.speak(callback);
        Debug.Log(callback);
    }
}


