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


public class inputChat : MonoBehaviour
{
    //對話窗口
    public GameObject chatWindow;
    //對話條
    public GameObject GptChatItem;
    //對話條
    public GameObject UserChatItem;

    //start 動畫
    private bool ready = false;
    public Animator balckfadeAni;
    public Animator loadingAni;
    
    //輸入送出按鍵
    //public Button yourButton;
    //用戶 輸入框
    //public InputField chatInput;
    //API Key 輸入框
    //public InputField OpenAI_Key;
    //chatGPT對話 物件
    public ChatGPT chatGPT;
    public bool isrecording = false;
    //線程鎖
    private object threadLocker = new object();
    //語音辨識等待值
    private bool waitingForReco;
    //語音辨識狀態 0:未開始辨識 1：輸入辨識中 2：辨識終止
    private int Rec = 0;

    //SpeechRecognizer 物件
    private SpeechRecognizer recognizer;
    //輸入語音訊息
    private string message;
    //  當前狀態
    private string now_emo, now_act, now_face;
    //上次回傳訊息

    private string now_time;
    private DateTime currenttime;
    //  當前狀態
    private string last_callback = "fjweiofwoanow;iefnoiwefnowfnowe";
    //AI暫停播放關鍵字
    private const string callAI = "暫停播放";

    List<string> aitalkingkeyword = new List<string> {"感謝回覆", "收到命令", "收到指令", "感謝您的回覆", "非常抱歉", "感謝回報"};
    List<string> aitalkingkeyword2 = new List<string> {"感謝回覆", "感謝您的回覆", "感謝回報"};
    //AI語音播放器
    private text_to_voice Speaker;
    //python控制
    //private PythonScript PythonScript;
    //設備關鍵字
    //private const string callEquipment = "操作設備";
    List<string> initialKeywords = new List<string> { "操作", "打開", "關閉", "關掉", "開啟", "切換", "電燈", "音樂", "冷氣", "暫停", "播放" };
    //設備模式
    private bool equipmentMode = false;
    //郵件模式
    private Gmail gmailObject = null;
    //觸發指令操作 (預設為進入設備模式 第一句話) 待指令集加入後改由偵測到指令集指令後觸發
    private bool firstEquipment = true;
    //語音辨識工具
    private Microsoft.CognitiveServices.Speech.SpeechConfig configuration;
    //建立TextAsset
    public TextAsset TxtFile;
    //指令選擇
    private Command_control Command_control = new Command_control();
    //用來存放文本內容
    private string Mytxt;
    [SerializeField]
    private AnimationControl animationControl;
    [SerializeField] public List<ApiKeyData> ApiKey = new List<ApiKeyData>();

    private string modePath = Application.platform == RuntimePlatform.WindowsPlayer || Application.platform == RuntimePlatform.WindowsEditor ? @"Assets\Python\speechRecognition\mode.txt" : @"Assets/Python/speechRecognition/mode.txt";
    private string outputPath = Application.platform == RuntimePlatform.WindowsPlayer || Application.platform == RuntimePlatform.WindowsEditor ? @"Assets\Python\speechRecognition\output.txt" : @"Assets/Python/speechRecognition/output.txt";
    private string readyPath = Application.platform == RuntimePlatform.WindowsPlayer || Application.platform == RuntimePlatform.WindowsEditor ? @"Assets\Python\speechRecognition\ready.txt" : @"Assets/Python/speechRecognition/ready.txt";

    //搜索引擎API key
    string serpapi_Key = "";
    public string zapier_Key = "";

    public bool isSpeaking = false;
    public string speak_style = "assistant";

    public string[] animationNamesToPlay = { "2", "7", "8", "9", "10", "11" }; // 指定要播放的動畫名稱列表
    public float minTimeBetweenAnimations = 10.0f; // 最小間隔時間
    public float maxTimeBetweenAnimations = 20.0f; // 最大間隔時間
    public bool inputvoice = false;

    //python 辨識說話的進程
    public Process speechRecognitionProcess = null;

    void talkProcess(Process process)
    {
        speechRecognitionProcess = process;
    }

    static string ConvertWindowsToMacOSPath(string windowsPath)
    {
    	if(Path.DirectorySeparatorChar == '/')
    	{
	        string fullPath = Path.GetFullPath(windowsPath);
    	    string macOSPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);
	        return macOSPath;
    	}else return windowsPath;
    }

    void Start()
    {
        File.WriteAllText(modePath, "1");

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
        //AI語音播放器
        Speaker = new text_to_voice(ApiKey[0].key, ApiKey[0].region);
        configuration = SpeechConfig.FromSubscription(ApiKey[1].key, ApiKey[1].region);
        chatGPT.setApiKey(ApiKey[2].key);
        serpapi_Key = ApiKey[3].key;
        zapier_Key = ApiKey[4].key;

        configuration.SpeechRecognitionLanguage = "zh-TW";
        //Button btn = yourButton.GetComponent<Button>();
        //btn.onClick.AddListener(TaskOnClick);

        // 初始化 SpeechRecognizer 物件
        //InitializeSpeechRecognizer();

        //讀取文本
        Mytxt = ((TextAsset)Resources.Load("instruction")).text;
        //測試
        //print(Mytxt);
        StartCoroutine(PlayAnimationEveryMinute());
        // 開始一個協程來監視Rec的值
        StartCoroutine(MonitorRec());

        //chatGPT(聊天) 預設角色
        chatGPT.m_DataList.Add(new SendData("system", "你是虛擬助手，清將使用者的文字輸入都當作是你聽到的，你可以回答任何問題，請將你所有的問字回覆都當作你說出來的話；同時也是一個可以控制設備AI，在接收命令時，只表示願意執行即可，等待後續輸入再根據(裝置狀態)做回應，若(裝置狀態)是失敗的，請根據狀態描述提示用戶可能的錯誤原因。我知道現在的時間和日期。"));
        chatGPT.m_DataList.Add(new SendData("system", "你要在結尾輸出該次對話的情緒及動作、表情在括弧中，輸出規則為（情緒、動作、表情），請使用數字編號回答，" +
            "情緒：1.深情、2.憤怒、3.助理、4.冷靜、5.聊天、6.快樂、7.客戶服務、8.不滿、9.恐懼、10.友好、11.溫柔、12.抒情、13.新聞廣播、14.詩歌朗誦、15.悲傷、16.嚴肅、17.害羞、18.沮喪；" +
            "動作：1.普通地站著、2.雙手前後擺動顯得感到有點無聊、3.抱胸用力跺腳非常生氣、4.快速微微正式鞠躬、5.身體傾斜很不正式的鞠躬、6.低頭踢腳有點難過、7.大力揮手、8.打哈欠以及伸懶腰、9.開心的晃動身體與手臂、10.伸展手臂、11.雀躍的搖晃手臂與身體、12.興奮的小跳；" +
            "表情：1.微笑、2.覺得好笑的笑臉（眼睛沒有完全閉起來，嘴巴也沒有張開）、3.生氣、4.哀傷、5.驚訝、6.覺得非常好笑的笑臉（眼睛完全閉起、嘴巴微微張開）；" +
            "範例：「（6、2、2）」、「（3、4、1）」"));
        chatGPT.m_DataList.Add(new SendData("system", "若使用者回傳的語句開頭有（裝置狀態），則將其內容簡單告知使用者，請勿回覆「了解，感謝您的回報」" +
            "範例：使用者：(裝置狀態)裝置已操作成功，裝置目前狀態:啟用。你要回答：裝置已經操作成功"));
        chatGPT.m_DataList.Add(new SendData("system", "請判斷用戶關於這個主題的對話是否結束，若對話結束請在句子、情緒及動作的結尾附上（End）" +
            "範例：「（End）」"));

        //chatGPT(設備) 預設角色
        chatGPT.e_DataList.Add(new SendData("system", Mytxt));

        //讀取現有人物設定
        try
        {
            var inputString = File.ReadAllText("CharacterSetting.json");
            if (!String.IsNullOrEmpty(inputString))
                chatGPT.m_DataList = JsonUtility.FromJson<Serialization<SendData>>(inputString).ToList();
        }
        //無人物設定 則創建新檔案並儲存
        catch (Exception e)
        {
            // 如果讀取文件失敗，創建一個新的文件
            using (var fs = File.Create("CharacterSetting.json"))
            {
                fs.Close(); // 確保文件被關閉和釋放
            }
            var outputString = JsonUtility.ToJson(new Serialization<SendData>(chatGPT.m_DataList));
            File.WriteAllText("CharacterSetting.json", outputString);
        }
        //讀取現有對話紀錄
        try
        {
            var inputString = File.ReadAllText("ChatHistory.json");
            if (!String.IsNullOrEmpty(inputString))
            {
                var dataListFromJson = JsonUtility.FromJson<Serialization<SendData>>(inputString).ToList();
        
                foreach(var data in dataListFromJson)
                {
                    chatGPT.AddNewRecord(data);
                }
            }
        }
        //無對話紀錄 則創建空紀錄檔案
        catch (Exception e)
        {
            using (var fs = File.Create("ChatHistory.json"))
            {
                fs.Close(); // 確保文件被關閉和釋放
            }
            //用設備的DataList來當作空的資料加入ChatHistory.json
            List<SendData> EmptyList = new List<SendData>();
            var outputString = JsonUtility.ToJson(new Serialization<SendData>(EmptyList));
            File.WriteAllText("ChatHistory.json", outputString);
        }

        //讀取現有設備操作紀錄
        try
        {
            var inputString = File.ReadAllText("EquipmentLog.json");
            if (!String.IsNullOrEmpty(inputString))
                chatGPT.e_DataList = JsonUtility.FromJson<Serialization<SendData>>(inputString).ToList();
        }
        //無設備操作 則創建空紀錄檔案
        catch (Exception e)
        {
            File.Create("EquipmentLog.json");
        }

    }

    void OnDestroy()
    {
        // 釋放 SpeechRecognizer 物件資源
        if (recognizer != null)
        {
            recognizer.Dispose();
            recognizer = null;
        }
        //speechRecognitionProcess.Kill();
    }

    private async void InitializeSpeechRecognizer()
    {

        lock (threadLocker)
        {
            //等待語音辨識(啟用)
            waitingForReco = true;
            //語音辨識開始
            Rec = 1;
        }

        // 初始化 SpeechRecognizer 物件
        recognizer = new SpeechRecognizer(configuration);

        //語音辨識結果紀錄
        string newMessage = string.Empty;

        //語音辨識回傳
        var result = await recognizer.RecognizeOnceAsync().ConfigureAwait(false);

        //確認語音辨識
        if (result.Reason == ResultReason.RecognizedSpeech)
        {
            newMessage = result.Text;
            Rec = 2;
            inputvoice = true;
            //newMessage = "打開電燈。";
            //呼叫字串比較，不是由AI回答，並且提到"操作設備"則進入設備模式
            /*
            if (!equipmentMode)
            {
                //任務進入設備模式 (之後會由分析用戶任務導向的方式控制 目前仍由偵測關鍵詞進入)
                //var check_EquipmentMode_call = ClassSim.MatchKeywordSim(callEquipment, newMessage);
                //if (check_EquipmentMode_call >= 0.4)
                KeywordComparer comparer = new KeywordComparer(initialKeywords);
                if(comparer.CompareWithKeywords(newMessage))
                {
                    equipmentMode = true;
                    firstEquipment = true;
                }
            }
            */
        }
        else
        {
            Rec = 0;
            return;
        }
        //強制結束播放（不用等回傳到）ChatGPT的時間
        var check_num1 = ClassSim.MatchKeywordSim(callAI, newMessage);
        if (check_num1 >= 0.5)
        {
            Rec = 0;
            //停止現有的AI對話與音輸出
            Speaker.Mute();
            return;
        }
        //呼叫字串比較，只要大於一定值就直接無視
        var check_num = ClassSim.MatchKeywordSim(last_callback, newMessage);
        if (check_num >= 0.5)
        {
            Rec = 0;
            return;
        }

        lock (threadLocker)
        {
            //傳輸輸入訊息
            message = newMessage;
            //等待語音辨識(停用)
            waitingForReco = false;
            //語音辨識完成
            Rec = 2;
        }

        // 設定回傳的結果類型為 RecognizedSpeech
        /*
        recognizer.Recognized += (s, e) =>
        {
            if (e.Result.Reason == ResultReason.RecognizedSpeech)
            {
                recognizedText = e.Result.Text;
                //chatInput.text = recognizedText;
                //recognizer.StopContinuousRecognitionAsync().ConfigureAwait(false);
            }
        };
        */

        // 開始語音辨識
        // await recognizer.StartContinuousRecognitionAsync().ConfigureAwait(false);

        //保证至少一个任务完成（等待到结束时间执行后再结束）
        //Task.WaitAny(new[] { stopRecognition.Task });
    }

    //當發送鍵被按下
    void TaskOnClick()
    {
        //設置ApiKey 已內建
        //chatGPT.setApiKey(OpenAI_Key.text);
        print(chatGPT.getApiKey());

        //判斷輸入框是否為空
        //if (!string.IsNullOrEmpty(chatInput.text))
        //    toSendData();
    }

    //GPT訊息 發送動作 (任務辨識)
    public void toSendData_T(string _msg)
    {
        //StartCoroutine(TurnToLastLine());
        //POST GPT訊息
        StartCoroutine(chatGPT.GetPostData_T(_msg, CallBack_T));
    }

    //GPT訊息 發送動作
    public void toSendData(string _msg)
    {

        //取得輸入訊息
        //string _msg = chatInput.text;
        print(_msg);
        //清空輸入框
        //chatInput.text = "";
        //建構對話條
        if (!equipmentMode)
        {
            /*
            var vChatWindow = chatWindow.transform.localPosition;
            var itemGround = Instantiate(UserChatItem, vChatWindow, Quaternion.identity);
            itemGround.transform.parent = chatWindow.transform;
            itemGround.transform.GetChild(1).transform.GetChild(0).GetComponent<Text>().text = _msg;
            checkChatsBoxToDelete();
            */
        }
        else if (equipmentMode && firstEquipment)
        {
            /*
            var vChatWindow = chatWindow.transform.localPosition;
            var itemGround = Instantiate(UserChatItem, vChatWindow, Quaternion.identity);
            itemGround.transform.parent = chatWindow.transform;
            itemGround.transform.GetChild(1).transform.GetChild(0).GetComponent<Text>().text = _msg;
            itemGround.transform.GetChild(1).GetComponent<Image>().color = new UnityEngine.Color(1, 0.8056f, 0.4332f, 0.684f);//0.0038
            checkChatsBoxToDelete();
            */
        }
        else if (equipmentMode && !firstEquipment)
        {
            equipmentMode = false;
        }
        //POST GPT訊息 (並添加訊息備註 提示chatGPT回答規範)
        if (equipmentMode && firstEquipment)
        {
            _msg += "(簡短回覆收到命令即可，勿確切告知執行與否)";
            firstEquipment = false;
        }
        //
        StartCoroutine(TurnToLastLine());
        //POST GPT訊息
        StartCoroutine(chatGPT.GetPostData(_msg, CallBack));
    }

    //GPT訊息 發送動作 (設備模式)
    public void toSendData_E(
        string _msg    //文字消息
    )
    {
        print(_msg);
        //
        //不與對話行交互 (無須印出對話)
        //
        StartCoroutine(TurnToLastLine());
        //POST GPT訊息 (並添加訊息備註 提示chatGPT回答規範)
        _msg += "(簡短回覆編號即可)";
        StartCoroutine(chatGPT.GetPostData_E(_msg, CallBack_E));
    }

    //GPT訊息 發送動作 (上網任務)
    public void toSendData_I(
        string _msg    //文字消息
    )
    {
        /*
        var vChatWindow = chatWindow.transform.localPosition;
        var itemGround = Instantiate(UserChatItem, vChatWindow, Quaternion.identity);
        itemGround.transform.parent = chatWindow.transform;
        itemGround.transform.GetChild(1).transform.GetChild(0).GetComponent<Text>().text = _msg;
        checkChatsBoxToDelete();
        */


        //添加對話紀錄 (紀錄Google搜尋結果)
        chatGPT.AddNewRecord(new SendData("user", _msg));

        StartCoroutine(PythonScript.Search(
            CallBack_I,
            "Search",
            chatGPT.getApiKey(),
            serpapi_Key,
            _msg));
    }

    //GPT訊息 發送動作 (Gmail)
    public void toSendData_G(
       string _msg    //文字消息
   )
    {
        /*
        var vChatWindow = chatWindow.transform.localPosition;
        var itemGround = Instantiate(UserChatItem, vChatWindow, Quaternion.identity);
        itemGround.transform.parent = chatWindow.transform;
        itemGround.transform.GetChild(1).transform.GetChild(0).GetComponent<Text>().text = _msg;
        checkChatsBoxToDelete();
        */

        //添加對話紀錄 (紀錄Google搜尋結果)
        chatGPT.AddNewRecord(new SendData("user", _msg));

        StartCoroutine(PythonScript.Search(
            CallBack_G,
            "gmail",
            chatGPT.getApiKey(),
            zapier_Key,
            _msg));
    }


    //GPT訊息 回傳動作 (網路查詢)
    void CallBack_I(string _callback)
    {
        print("search:" + _callback);

        //添加對話紀錄 (紀錄Google搜尋結果)
        chatGPT.AddNewRecord(new SendData("assistant", _callback));
        //chatGPT狀態 (空閒)
        chatGPT.taskState = 0;
        //建構對話條
        var itemGround = Instantiate(GptChatItem, chatWindow.transform);
        // 設置初始縮放為零
        itemGround.transform.localScale = Vector3.zero;
        // 獲取對話框的RectTransform组件
        RectTransform dialogRect = itemGround.GetComponent<RectTransform>();
        // 設置對話框文字內容
        Text dialogText = itemGround.transform.GetChild(0).transform.GetChild(0).GetComponent<Text>();
        dialogText.text = _callback;
        UnityEngine.Debug.Log("時間"+System.DateTime.Now);
        // 動畫
        dialogRect.DOScale(Vector3.one, 0.5f).SetEase(Ease.OutBack);
        checkChatsBoxToDelete();
        //AI語音播放
        speak(_callback);
        //
        StartCoroutine(TurnToLastLine());


        //存取現有對話紀錄
        //var outputString = JsonUtility.ToJson(new Serialization<SendData>(chatGPT.m_DataList));
        //try
        //{
        //    File.WriteAllText("MyFile.json", outputString);
        //}
        //無對話紀錄 則創建空紀錄檔案
        //catch (Exception e)
        //{
        //    File.Create("MyFile.json");
        //    File.WriteAllText("MyFile.json", outputString);
        //}
        //重新計時
        timer = 0f;
        countingStop = false;
    }

    //GPT訊息 回傳動作
    public void CallBack_G(string _callback)
    {
        print("send:" + _callback);

        //添加對話紀錄
        //chatGPT.m_DataList.Add(new SendData("assistant", _callback));
        //chatGPT狀態 (空閒)
        //chatGPT.taskState = 0;
        //建構對話條
        var itemGround = Instantiate(GptChatItem, chatWindow.transform);
        // 設置初始縮放為零
        itemGround.transform.localScale = Vector3.zero;
        // 獲取對話框的RectTransform组件
        RectTransform dialogRect = itemGround.GetComponent<RectTransform>();
        // 設置對話框文字內容
        Text dialogText = itemGround.transform.GetChild(0).transform.GetChild(0).GetComponent<Text>();
        dialogText.text = _callback;
        UnityEngine.Debug.Log("時間"+System.DateTime.Now);
        // 動畫
        dialogRect.DOScale(Vector3.one, 0.5f).SetEase(Ease.OutBack);
        checkChatsBoxToDelete();
        //AI語音播放
        speak(_callback);
        //
        StartCoroutine(TurnToLastLine());


        //存取現有對話紀錄
        //var outputString = JsonUtility.ToJson(new Serialization<SendData>(chatGPT.m_DataList));
        //try
        //{
        //    File.WriteAllText("MyFile.json", outputString);
        //}
        //無對話紀錄 則創建空紀錄檔案
        //catch (Exception e)
        //{
        //    File.Create("MyFile.json");
        //    File.WriteAllText("MyFile.json", outputString);
        //}
        SendMailDone();
    }

     public void toSendData_time(string _msg)
    {

        DateTime currenttime=System.DateTime.Now;
        _msg += "現在的時間是"+currenttime+"，請簡單回答";
        UnityEngine.Debug.Log("11時間"+currenttime);
        //取得輸入訊息309
        //string _msg = chatInput.text;
        print(_msg);
        //清空輸入框
        //chatInput.text = "";
        //建構對話條

        StartCoroutine(TurnToLastLine());
        //POST GPT訊息
        StartCoroutine(chatGPT.GetPostData(_msg, CallBack));
    }


    //GPT訊息 回傳動作 (任務辨識)
    private void CallBack_T(string _callback, string originalText)
    {
        int task = 0;

        try
        {
            task = int.Parse(Regex.Replace(_callback, @"\D", string.Empty));
        }
        catch
        {
            task = 0;
        }

        if (task == 4)
        { }
        else if (task == 2)
        {
            //建構對話條
            var itemGround = Instantiate(UserChatItem, chatWindow.transform);
            // 設置初始縮放為零
            itemGround.transform.localScale = Vector3.zero;
            // 獲取對話框的RectTransform组件
            RectTransform dialogRect = itemGround.GetComponent<RectTransform>();
            // 設置對話框文字內容
            Text dialogText = itemGround.transform.GetChild(1).transform.GetChild(0).GetComponent<Text>();
            dialogText.text = originalText;
            itemGround.transform.GetChild(1).GetComponent<Image>().color = new UnityEngine.Color(1, 0.8056f, 0.4332f, 0.684f);//0.0038
            // 動畫
            dialogRect.DOScale(Vector3.one, 0.5f).SetEase(Ease.OutBack);
            UnityEngine.Debug.Log("時間"+System.DateTime.Now);
            checkChatsBoxToDelete();
        }
        else
        {
            //建構對話條
            var itemGround = Instantiate(UserChatItem, chatWindow.transform);
            // 設置初始縮放為零
            itemGround.transform.localScale = Vector3.zero;
            // 獲取對話框的RectTransform组件
            RectTransform dialogRect = itemGround.GetComponent<RectTransform>();
            // 設置對話框文字內容
            Text dialogText = itemGround.transform.GetChild(1).transform.GetChild(0).GetComponent<Text>();
            dialogText.text = originalText;
            // 動畫
            dialogRect.DOScale(Vector3.one, 0.5f).SetEase(Ease.OutBack);
            checkChatsBoxToDelete();
        }

        UnityEngine.Debug.Log("task: " + task + " " + originalText);
        chatGPT.taskQueue.Enqueue(new SendQueue(originalText, task));

    }

    //GPT訊息 回傳動作
    private void CallBack(string _callback, string sendMessage, string emotion, bool isEnd)
    {
        //取得回傳訊息
        _callback = _callback.Trim();
        print("M: " + _callback);
        print("emotion: " + emotion);
        // 提取字串中的數字
        var matches = Regex.Matches(emotion, @"\d+");
        if (matches.Count >= 3)
        {
            now_emo = matches[0].Value;
            now_act = matches[1].Value;
            now_face = matches[2].Value;
            print($"now_emo: {now_emo}");
            print($"now_act: {now_act}");
            print($"now_face: {now_face}");
        }
        if ((isEnd && !equipmentMode) || Regex.Match(sendMessage,@"\(裝置狀態\)").Success)
        {
            print($"ChatGPT - 判斷對話結束");
            //重新計時
            timer = 0f;
            countingStop = false;
        }
        //根據回傳的數字決定音調、表情、動作
        Speaker.ChangeEmotion(now_emo);
        animationControl.set_action(now_act);
        animationControl.set_face(now_face);
        last_callback = _callback;

        if (equipmentMode)
            toSendData_E(sendMessage);
        else
            //chatGPT狀態 (空閒)
            chatGPT.taskState = 0;

        //建構對話條
        var itemGround = Instantiate(GptChatItem, chatWindow.transform);
        // 設置初始縮放為零
        itemGround.transform.localScale = Vector3.zero;
         // 獲取對話框的RectTransform组件
        RectTransform dialogRect = itemGround.GetComponent<RectTransform>();
        // 設置對話框文字內容
        Text dialogText = itemGround.transform.GetChild(0).transform.GetChild(0).GetComponent<Text>();
        dialogText.text = _callback;
        // 動畫
        dialogRect.DOScale(Vector3.one, 0.5f).SetEase(Ease.OutBack); 
        UnityEngine.Debug.Log("時間"+System.DateTime.Now);
        checkChatsBoxToDelete();
        //AI語音播放
        speak(_callback);
        //
        StartCoroutine(TurnToLastLine());


        //存取現有對話紀錄
        DateTime currenttime=System.DateTime.Now;
        var outputString = JsonUtility.ToJson(new Serialization<SendData>(chatGPT.m_DataList));
       
     
       
       
        UnityEngine.Debug.Log("309時間"+currenttime);
        //var outputString1 = JsonUtility.ToJson(currenttime);
        //try
        //{
        //    File.WriteAllText("MyFile.json", outputString);
        //}
        //無對話紀錄 則創建空紀錄檔案
        //catch (Exception e)
        //{
        //    File.Create("MyFile.json");
        //    File.WriteAllText("MyFile.json", outputString);
        //}
    }

    //GPT訊息 回傳動作 (設備操作使用)
    private void CallBack_E(string _callback)
    {
        //取得回傳訊息
        _callback = _callback.Trim();
        print("E: " + _callback);
        String equipmentState;
        if (Command_control.choose_command(_callback))
        {
            StartCoroutine(TurnToLastLine());
            equipmentState = "(裝置狀態)裝置已操作成功，裝置目前狀態:啟用";
        }
        else
        {
            StartCoroutine(TurnToLastLine());
            equipmentState = "(裝置狀態)查無此裝置";
        }
        //觸發指令操作
        //
        //實際裝置控制指令
        //
        //根據裝置狀態讓chatGPT(聊天)回應
        //String equipmentState = "(裝置狀態)裝置已操作成功，裝置目前狀態:啟用";
        //String equipmentState = "(裝置狀態)裝置無法連接";
        //String equipmentState = "(裝置狀態)查無此裝置";
        //String equipmentState = "(裝置狀態)操作失敗，原因:未知";
        toSendData(equipmentState);

        //讀取現有對話紀錄
        var outputString = JsonUtility.ToJson(new Serialization<SendData>(chatGPT.e_DataList));
        try
        {
            File.WriteAllText("EquipmentLog.json", outputString);
        }
        //無對話紀錄 則創建空紀錄檔案
        catch (Exception e)
        {
            File.Create("EquipmentLog.json");
            File.WriteAllText("EquipmentLog.json", outputString);
        }
    }

    public void checkChatsBoxToDelete()
    {
        if (chatWindow.transform.childCount > 5)
        {
            Destroy(chatWindow.transform.GetChild(0).gameObject);
        }
    }

    private IEnumerator TurnToLastLine()
    {
        yield return new WaitForEndOfFrame();
    }

    public void Quit()
    {
        //speechRecognitionProcess.Kill();
        Application.Quit();
    }

    public void SendMailDone()
    {
        chatGPT.taskState = 0;
        gmailObject = null;

        //重新計時
        timer = 0f;
        countingStop = false;
    }

    public void CancelSendMail()
    {
        chatGPT.taskState = 0;
        gmailObject = null;
        String text = "好的。已取消寄送Email。";
        //建構對話條
        var itemGround = Instantiate(GptChatItem, chatWindow.transform);
        // 設置初始縮放為零
        itemGround.transform.localScale = Vector3.zero;
        // 獲取對話框的RectTransform组件
        RectTransform dialogRect = itemGround.GetComponent<RectTransform>();
        // 設置對話框文字內容
        Text dialogText = itemGround.transform.GetChild(0).transform.GetChild(0).GetComponent<Text>();
        dialogText.text = text;
        // 動畫
        dialogRect.DOScale(Vector3.one, 0.5f).SetEase(Ease.OutBack);
        checkChatsBoxToDelete();
        //AI語音播放
        speak(text);

        //重新計時
        timer = 0f;
        countingStop = false;
    }

    public void CancelMailMode()
    {
        chatGPT.taskState = 0;
        gmailObject = null;
        String text = "好的。";
        //建構對話條
        var itemGround = Instantiate(GptChatItem, chatWindow.transform);
        // 設置初始縮放為零
        itemGround.transform.localScale = Vector3.zero;
        // 獲取對話框的RectTransform组件
        RectTransform dialogRect = itemGround.GetComponent<RectTransform>();
        // 設置對話框文字內容
        Text dialogText = itemGround.transform.GetChild(0).transform.GetChild(0).GetComponent<Text>();
        dialogText.text = text;
        // 動畫
        dialogRect.DOScale(Vector3.one, 0.5f).SetEase(Ease.OutBack);
        checkChatsBoxToDelete();
        //AI語音播放
        speak(text);

        //重新計時
        timer = 0f;
        countingStop = false;
    }

    public void CancelChangeMailBox()
    {
        chatGPT.taskState = 0;
        gmailObject = null;
        String text = "好的。";
        //建構對話條
        var itemGround = Instantiate(GptChatItem, chatWindow.transform);
        // 設置初始縮放為零
        itemGround.transform.localScale = Vector3.zero;
        // 獲取對話框的RectTransform组件
        RectTransform dialogRect = itemGround.GetComponent<RectTransform>();
        // 設置對話框文字內容
        Text dialogText = itemGround.transform.GetChild(0).transform.GetChild(0).GetComponent<Text>();
        dialogText.text = text;
        // 動畫
        dialogRect.DOScale(Vector3.one, 0.5f).SetEase(Ease.OutBack);
        checkChatsBoxToDelete();
        //AI語音播放
        speak(text);

        //重新計時
        timer = 0f;
        countingStop = false;
    }

    private Gmail.MailBoxData selectLestMailBox()
    {
        List<SendData> mb_DataList = new List<SendData>();
        int selectLest = -1;

        //讀取現有信箱紀錄
        try
        {
            var inputString = File.ReadAllText("EmailBox_Keys.json");
            if (!String.IsNullOrEmpty(inputString))
                mb_DataList = JsonUtility.FromJson<Serialization<SendData>>(inputString).ToList();
        }
        //無信箱紀錄 則創建空信箱紀錄
        catch (Exception e)
        {
            File.Create("EmailBox_Keys.json");
            File.WriteAllText("EmailBox_Keys.json",
                @"{""target"":[{""role"":""system"",""content"":""你是一個密鑰管理資料庫請根據用戶輸入的文本判斷要調取何個資料庫。並將訊息原封不動輸出。""}]}"
                );
        }

        try
        {
            var inputString = File.ReadAllText("MailSelect.txt");
            selectLest = int.Parse(inputString);
        }
        //無信箱紀錄 則創建空信箱紀錄
        catch (Exception e)
        {
            File.WriteAllText("MailSelect.txt", @"1");
        }

        try
        {
            string name = Regex.Replace(
                Regex.Match(
                    mb_DataList[selectLest].content,
                    @"\S+的信箱"
                ).Value,
                @"的信箱", string.Empty
            );
            string key = Regex.Replace(
                Regex.Match(
                    mb_DataList[selectLest].content,
                    @"(（\S+）)|(\(\S+\))$"
                ).Value,
                @"((編號\d+：)|\(|\)|（|）)", string.Empty
            );
            print($"name:{name},key:{key}");
            return new Gmail.MailBoxData(name, key);
        }
        //無信箱紀錄
        catch (Exception e)
        {
            chatGPT.taskState = 0;
            gmailObject = null;
            String text = "尚未設置好信箱權限，故無法執行此操作。";
            //建構對話條
            var itemGround = Instantiate(GptChatItem, chatWindow.transform);
            // 設置初始縮放為零
            itemGround.transform.localScale = Vector3.zero;
            // 獲取對話框的RectTransform组件
            RectTransform dialogRect = itemGround.GetComponent<RectTransform>();
            // 設置對話框文字內容
            Text dialogText = itemGround.transform.GetChild(0).transform.GetChild(0).GetComponent<Text>();
            dialogText.text = text;
            // 動畫
            dialogRect.DOScale(Vector3.one, 0.5f).SetEase(Ease.OutBack);
            checkChatsBoxToDelete();
            //AI語音播放
            speak(text);
            return null;
        }
    }

    void Update()
    {
        if (!ready)
        {
            try
            {
                var str = File.ReadAllText(readyPath);
                if (!String.IsNullOrEmpty(str))
                {
                    ready = true;
                    balckfadeAni.SetBool("ready", true);
                    loadingAni.SetBool("ready", true);
                }
            }
            catch { }
            return;
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            //speechRecognitionProcess.Kill();
            Application.Quit();
        }

        lock (threadLocker)
        {
            //語音辨識完成
            if (Rec == 0)
            {
                Rec = 1;
                try
                {
                    var strData = File.ReadAllText(outputPath);
                    if (!String.IsNullOrEmpty(strData))
                    {
                        File.WriteAllText(outputPath, string.Empty);
                        Match m = Regex.Match(strData, @"\[\'.+\'\]", RegexOptions.IgnoreCase);
                        if (m.Success)
                        {
                            //重新計時
                            timer = 0f;
                            string toMessage = Regex.Replace(m.Value, @"[\[,',\]]", string.Empty);
                            //print(text);
                            //呼叫字串比較，只要大於一定值就直接無視
                            var check_num = ClassSim.MatchKeywordSim(last_callback, toMessage);
                            KeywordComparer check_num3 = new KeywordComparer();
                            check_num3.KeywordComparer2(aitalkingkeyword);
                            var ismatch = check_num3.CompareWithKeywords(toMessage);
                            if (check_num >= 0.5 || ismatch)
                            {
                                Rec = 0;
                                File.WriteAllText(outputPath, string.Empty);
                                return;
                            }
                            else{
                                //停止現有的AI對話與音輸出
                                Speaker.Mute();

                                //強制結束播放（不用等回傳到）ChatGPT的時間
                                var check_num1 = ClassSim.MatchKeywordSim(callAI, toMessage);
                                if (check_num1 >= 0.5)
                                {
                                    Rec = 0;
                                    //停止現有的AI對話與音輸出
                                    Speaker.Mute();
                                    return;
                                }

                                //chatGPT請求 (做任務分類)
                                if (gmailObject == null)
                                {   
                                    //暫停計時
                                    countingStop = true;
                                    toSendData_T(toMessage);
                                }
                                else
                                    gmailObject.toSendData(toMessage);
                            }
                        }
                    }
                }
                catch { }
                Rec = 0;
            }
        }

        /*
        //控制語音辨識線程
        lock (threadLocker)
        {
            //語音辨識完成
            if (Rec == 2)
            {
                Rec = 0;
                //一般聊天
                if (!string.IsNullOrEmpty(message))
                {
                    string toMessage = message;
                    //淨空傳遞訊息
                    message = string.Empty;
                    //停止現有的AI對話與音輸出
                    Speaker.Mute();
                    //輸入框顯示本次輸入的訊息
                    //chatInput.text = message;
                    //chatGPT請求 (做任務分類)
                    if (gmailObject == null)
                        toSendData_T(toMessage);
                    else
                        gmailObject.toSendData(toMessage);
                }
            }
            //啟用語音辨識
            if (Rec == 0)
            {
                InitializeSpeechRecognizer();
            }
        }
        */
        lock (chatGPT.threadLocker)
        {
            if (chatGPT.taskState == 0 && chatGPT.taskQueue.Count > 0)
            {
                chatGPT.taskState = 1;

                SendQueue task = chatGPT.taskQueue.Dequeue();
                string originalText = task.getText();
                int taskClass = task.getTaskClass();

                switch (taskClass)
                {
                    case 2: //設備操作
                        equipmentMode = true;
                        firstEquipment = true;
                        toSendData(originalText);
                        break;
                    case 4:
                        Gmail.MailBoxData mailBoxData = selectLestMailBox();
                        if (mailBoxData == null)
                            break;
                        gmailObject = new Gmail(chatGPT, this, mailBoxData);
                        gmailObject.toSendData(originalText);
                        //toSendData_G(originalText);
                        UnityEngine.Debug.Log("傳送郵件");
                        break;
                    case 5:
                        toSendData_time(originalText);
                        break;
                        
                    case 3: //時效性問答 or 網路查詢
                            //print("需使用網路查詢");
                        toSendData_I(originalText);
                        break;

                    case 1: //一般聊天 or 非時效性問答
                    default:
                        toSendData(originalText);
                        break;
                }
            }
        }
    }
    public void speak(string text)
    {
        // 將語音合成標誌設置為 true，表示正在進行語音合成
        isSpeaking = true;

        File.WriteAllText(modePath, "2");

        // 開始執行 Coroutine，用於動畫的重複執行
        StartCoroutine(AnimateFace());

        // 調用 readString 方法來進行語音合成，並在合成完成後設置 isSpeaking 為 false
        Speaker.readString(text, speak_style, () =>
        {
            isSpeaking = false;
            File.WriteAllText(modePath, "1");
        });
    }

    private IEnumerator AnimateFace()
    {
        int i = 1;
        while (isSpeaking)
        {
            if (i == 1) yield return new WaitForSeconds(1.8f);
            yield return new WaitForSeconds(0.15f);
            animationControl.set_face("6");
            // 時間間隔
            yield return new WaitForSeconds(0.3f);
            animationControl.set_face("1");
            i++;

        }

    }
    private IEnumerator PlayAnimationEveryMinute()
    {
        while (isSpeaking == false)
        {
            int randomIndex = UnityEngine.Random.Range(0, animationNamesToPlay.Length);
            string randomAnimation = animationNamesToPlay[randomIndex];

            float randomTime = UnityEngine.Random.Range(minTimeBetweenAnimations, maxTimeBetweenAnimations);
            yield return new WaitForSeconds(randomTime);
            // 播放動畫
            animationControl.set_action(randomAnimation);


        }
    }

    private bool countingStop = false; // 計時暫停
    private bool isCounting = true; // 用來檢測是否正在計時
    private float timer = 0f; // 計時器
    IEnumerator MonitorRec()
    {
        while (isCounting)
        {
            yield return new WaitForSeconds(1f); // 每秒等待一次

            if (!countingStop && ready)
            {
                timer += 1f; // 計時遞增

                if (timer >= 60f && inputvoice == false)
                {
                    SceneManager.LoadScene(0);
                    isCounting = false;
                }
            }
        }
    }
}

//list轉存JSON
[Serializable]
public class Serialization<T>
{
    [SerializeField]
    List<T> target;
    public List<T> ToList() { return target; }

    public Serialization(List<T> target)
    {
        this.target = target;
    }
}

// API金鑰 資料結構
[Serializable]
public class ApiKeyData
{
    public string key; // 使用模型
    public string region; // 對話紀錄
}

//字串比較演算法
public class ClassSim
{
    public static double MatchKeywordSim(string keyword, string matchkeyword)
    {
        List<char> keywordList = keyword.ToCharArray().ToList();
        List<char> matchkeywordList = matchkeyword.ToCharArray().ToList();
        List<char> unionKeyword = keywordList.Union(matchkeywordList).ToList<char>();
        List<int> arrA = new List<int>();
        List<int> arrB = new List<int>();
        foreach (var str in unionKeyword)
        {
            arrA.Add(keywordList.Where(x => x == str).Count());
            arrB.Add(matchkeywordList.Where(x => x == str).Count());
        }
        double num = 0;
        double numA = 0;
        double numB = 0;
        for (int i = 0; i < unionKeyword.Count; i++)
        {
            num += arrA[i] * arrB[i];
            numA += Math.Pow(arrA[i], 2);
            numB += Math.Pow(arrB[i], 2);
        }
        double cos = num / (Math.Sqrt(numA) * Math.Sqrt(numB));
        return cos;
    }
}

//keyword的物件
public class KeywordComparer
{
    private List<string> keywords;

    public void KeywordComparer1(List<string> initialKeywords)
    {
        keywords = initialKeywords;
    }
    public void KeywordComparer2(List<string> aitalkingkeyword)
    {
        keywords = aitalkingkeyword;
    }
    //用於一一比較
    public bool CompareWithKeywords(string matchKeyword)
    {
        foreach (string keyword in keywords)
        {
            double similarity = ClassSim.MatchKeywordSim(keyword, matchKeyword);
            if (similarity > 0.4)
            {
                return true;
            }
        }

        return false;
    }
}
