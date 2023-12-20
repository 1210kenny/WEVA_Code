using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using UnityEngine;
using UnityEngine.UI;
using Microsoft.CognitiveServices.Speech;
using static ChatGPT;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Diagnostics;
//using UnityEditor.VersionControl;
using static UnityEngine.UI.Image;
using System.Threading;
using UnityEngine.SceneManagement;
using System.Timers;
using static Gmail;
using DG.Tweening;
using Newtonsoft.Json;

public class start_object : MonoBehaviour
{
    private string modePath = Application.platform == RuntimePlatform.WindowsPlayer || Application.platform == RuntimePlatform.WindowsEditor ? @"Assets\Python\speechRecognition\mode.txt" : @"Assets/Python/speechRecognition/mode.txt";
    private string outputPath = Application.platform == RuntimePlatform.WindowsPlayer || Application.platform == RuntimePlatform.WindowsEditor ? @"Assets\Python\speechRecognition\output.txt" : @"Assets/Python/speechRecognition/output.txt";
    private string readyPath = Application.platform == RuntimePlatform.WindowsPlayer || Application.platform == RuntimePlatform.WindowsEditor ? @"Assets\Python\speechRecognition\ready.txt" : @"Assets/Python/speechRecognition/ready.txt";
    [SerializeField] public List<ApiKeyData> ApiKey = new List<ApiKeyData>();

    //python 辨識說話的進程
    public Process speechRecognitionProcess = null;
    public static start_object Instance { get; private set; }
    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else if (Instance != this)
        {
            Destroy(gameObject);
        }
    }

    void Start(){
        DontDestroyOnLoad(gameObject);
        try
        {
            var outputString = "First";
            File.WriteAllText("StartOrNot.json", outputString);
        }
        catch
        {
            // 如果讀取文件失敗，創建一個新的文件
            using (var fs = File.Create("StartOrNot.json"))
            {
                fs.Close(); // 確保文件被關閉和釋放
            }
            var outputString = "First";
            File.WriteAllText("StartOrNot.json", outputString);
        }

        //讀取鑰匙
        try
        {
            var inputString = File.ReadAllText("Keys.json");
            ApiKey = JsonUtility.FromJson<Serialization<ApiKeyData>>(inputString).ToList();
        }
        //無鑰匙 回報
        catch (Exception e)
        {
            print("No have Key file.");
        }
        
        //清除已存在語音輸出文件中的資料
        File.WriteAllText(readyPath, string.Empty);
        File.WriteAllText(outputPath, string.Empty);
        //python 辨識說話的進程
        StartCoroutine(PythonScript.speechRecognition(talkProcess, ApiKey[1].key, ApiKey[1].region));
    }
    
    void Update(){
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            speechRecognitionProcess.Kill();
            Application.Quit();
        }
    }
    void OnDestroy(){
        try
        {
            var outputString = "First";
            File.WriteAllText("StartOrNot.json", outputString);
        }
        catch
        {
            // 如果讀取文件失敗，創建一個新的文件
            using (var fs = File.Create("StartOrNot.json"))
            {
                fs.Close(); // 確保文件被關閉和釋放
            }
            var outputString = "First";
            File.WriteAllText("StartOrNot.json", outputString);
        }
        if (speechRecognitionProcess != null && !speechRecognitionProcess.HasExited)
        {
            speechRecognitionProcess.Kill();
            speechRecognitionProcess = null;
        }
    }
    public void Quit()
    {
        speechRecognitionProcess.Kill();
        Application.Quit();
    }
    void talkProcess(Process process)
    {
        speechRecognitionProcess = process;
    }
}
