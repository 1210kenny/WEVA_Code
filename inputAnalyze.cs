using Microsoft.CognitiveServices.Speech;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static ChatGPT;
using UnityEngine.Networking;
using System;
using Unity.VisualScripting;
using System.Text.RegularExpressions;

public class inputAnalyze : MonoBehaviour
{
    //AI API_url
    private const string m_ApiUrl = "https://www.clueai.cn/modelfun/api/serving_api";
    [SerializeField]
    private AnimationControl animationControl;

    // Send 資料結構 
    [Serializable]
    public class SendData
    {
        public string task_type = "classify";
        public string task_name; //AI 功能 見:https://www.clueai.cn/
        public string[] input_data; //輸入文本
        public string[] labels = { "喜", "怒", "哀", "樂", "积极", "消极" };//情緒分類(可自訂)
        public SendData() { }
        public SendData(string function, string input)
        {
            task_name = function; //AI 功能
            input_data = new string[]{input}; //輸入文本
        }
    }

    // GPT返回訊息 資料結構 
    [Serializable]
    private class RequestBody
    {
        public string detail;
        public bool state;
        public List<MessageBack> result;
    }

    // GPT返回訊息 資料結構 
    [Serializable]
    private class MessageBack
    {
        public string input;
        public string prediction;
        public List<Confidence> confidence;
    }

    // GPT返回訊息 資料結構 
    [Serializable]
    private class Confidence
    {
        public string label;
        public double confidence;
    }

    public static IEnumerator GetPostData(
        string text,
        System.Action<string, string> _callback //異步回傳函式
        )
    {
        //建構 WebRequest POST
        using (UnityWebRequest request = new UnityWebRequest(m_ApiUrl, "POST"))
        {
            string _jsonText;
            try
            {
                _jsonText = JsonUtility.ToJson(new SendData("情感分析", ChineseProgram.ToSimplifiedChinese(text)));
            }
            catch
            {
                _jsonText = JsonUtility.ToJson(new SendData("情感分析", text));
            }

            //轉存格式
            byte[] data = System.Text.Encoding.UTF8.GetBytes(_jsonText);
            //導入request頭部資訊
            request.uploadHandler = (UploadHandler)new UploadHandlerRaw(data);
            request.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.SetRequestHeader("Model-name", "clueai-base");

            _callback(text, "");
            yield return null;
            /*
            //發送
            yield return request.SendWebRequest();

            //後台回報 回傳碼 (200為成功)
            print(request.responseCode);
            if (request.responseCode == 200)
            {
                _callback(text, "");

                //取出回傳訊息
                string _msg = request.downloadHandler.text;
                //_msg.Replace('\'', '\"');
                RequestBody _textback = JsonUtility.FromJson<RequestBody>(_msg);
                if (_textback != null)
                {
                    string _backMsg = _textback.result[0].prediction;
                    
                    //返回函式
                    _callback(text, _backMsg);
                }
            }
            */
        }
    }

    public static IEnumerator chatGPT_mood(
        string text,
        string text1,
        System.Action<string,string, string, bool> _callback //異步回傳函式
        )
    {
        bool end = Regex.IsMatch(text, @"(（End）)|(\(End\))$");
        text = Regex.Replace(text, @"((\(End\))|(（End）))", string.Empty);
        string mood = Regex.Replace(Regex.Match(text, @"(（[、,\d]+）)|(\([\,,\d]+\))$").Value, @"(\(|\)|（|）)", string.Empty);
        string mainText = Regex.Replace(text, @"(（\S+）)|(\(\S+\))$", string.Empty);
        _callback(mainText, text1, mood, end);
        yield return null;
    }
}
