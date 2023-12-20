using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;

public class Gmail
{
    //chatGPT對話 物件
    public ChatGPT chatGPT;
    public inputChat inputChat;

    private MailBoxData MailBox = new MailBoxData();
    private MailBody MailData = new MailBody();
    private int guideTask = 0;

    public bool threadLocker = false;

    // GPT返回訊息 MailBox 資料結構 
    [Serializable]
    public class MailBoxData
    {
        public string Name;
        public string Key;

        public MailBoxData()
        {
        }
        public MailBoxData(string Name, string Key)
        {
            this.Name = Name;
            this.Key = Key;
        }
    }

    // GPT返回訊息 本體 資料結構 
    [Serializable]
    private class MailBody
    {
        public int task = -1;
        public string Do;
        public string addressee;
        public string addressee_Email;
        public string subject;
        public string mailText;
    }

    public Gmail(ChatGPT chatGPT, inputChat inputChat, MailBoxData MailBox)
    {
        this.chatGPT = chatGPT;
        this.inputChat = inputChat;
        this.MailData = new MailBody();
        this.guideTask = 0;
        this.threadLocker = false;

        this.MailBox = MailBox;
    }

    public void toSendData(string _msg)
    {
        if (!threadLocker)
        {
            threadLocker = true;
            UserSpack(_msg);
            //POST GPT訊息
            switch (guideTask)
            {
                case 0:
                    inputChat.StartCoroutine(chatGPT.GetPostData_M(firstTaskClassification(_msg), CallBack_M_First));
                    break;
                case 1:
                    inputChat.StartCoroutine(chatGPT.GetPostData_M(replenishTaskClassification(_msg), CallBack_M_Replenish));
                    break;
                case 2:
                    inputChat.StartCoroutine(chatGPT.GetPostData_M(ConfirmTaskClassification(_msg), CallBack_M_Replenish));
                    break;
                case 6:
                    inputChat.StartCoroutine(chatGPT.GetPostData_M(ChangeMailBox(_msg), CallBack_M_ChangeMailBox));
                    break;
            }
        }
        return;
    }

    private List<ChatGPT.SendData> firstTaskClassification(string Text)
    {
        List<ChatGPT.SendData> m_DataList = new List<ChatGPT.SendData>();
        m_DataList.Add(new ChatGPT.SendData("system",
            "我是一個郵件任務分析器，請根據用戶輸入之文句，分析用戶之目的，並根據下列要求分類任務；" +
            "若用戶在任務中有告知其他寄件、查詢訊息，請使用括弧分開並依順序紀錄：" +
            "1.查詢郵件（查詢目標）" +
            "2.寄送郵件（郵件標題；收件人；郵件內容）" +
            "6.切換郵件、信箱帳戶（名稱）" +
            "5.中止寄信或查找信件" +

            "下列為範例：" +
            "「最近一封郵件由誰寄發給我的，並且節錄標題」，回答：「1（最新一封郵件寄件人以及信件標題）」；" +
            "「請幫我寄信給小明，有關會議記錄的資料」，回答：「2（有關會議記錄的資料；小明；）」；" +
            "「請幫我寄信，有關商品特價目錄，今晚會由依晨寄送至指定單位，並由單位批准。」，回答：「2（有關商品特價目錄；；今晚會由依晨寄送至指定單位，並由單位批准）」；" +
            "「請幫我寄信，標題：成績問題查詢，收件人：富城主任。」，回答：「2（成績問題查詢；富城主任；）」；" +
            "「寄信，第一階段的簽收核對帳本何時送達。」，回答：「2（；；第一階段的簽收核對帳本何時送達）」；" +
            "「我不需要寄信了」，回答：「5」；" +
            "「不用幫我寄信」，回答：「5」；" +
            "「取消寄信」，回答：「5」；" +
            "「我不需要查找信件了」，回答：「5」；" +
            "「不用幫我找信」，回答：「5」；" +
            "「取消找郵件」，回答：「5」；"+
            "「幫我切換到小花信箱」，回答：「6（小花）」；" +
            "「變更到富城的信箱帳號」，回答：「6（富城）」；" +
            "「幫我更換帳號」，回答：「6（）」；"
            ));

        //緩存發送的訊息
        m_DataList.Add(new ChatGPT.SendData("user", Text));
        return m_DataList;
    }

    private List<ChatGPT.SendData> replenishTaskClassification(string Text)
    {
        List<ChatGPT.SendData> m_DataList = new List<ChatGPT.SendData>();
        m_DataList.Add(new ChatGPT.SendData("system",
            "我是一個郵件任務分析器，請根據用戶輸入之文句，分析用戶之目的，並根據下列要求分類任務；" +
            "若用戶在任務中有告知其他寄件、查詢訊息，請使用括弧分開並依順序紀錄："+
            "1.郵件標題（郵件標題）" +
            "2.收件人（收件人）" +
            "3.郵件內容（郵件內容）" +
            "5.中止寄信" +

            "下列為範例：" +
            "「標題是會議記錄的資料」，回答：「1（會議記錄的資料）」；" +
            "「信件內容是今晚會由依晨寄送至指定單位，並由單位批准。」，回答：「3（今晚會由依晨寄送至指定單位，並由單位批准）」；" +
            "「我要寄給富城主任。」，回答：「2（富城主任）」；" +
            "「由AI節錄信件內容編寫標題」，回答：「1（由AI節錄信件內容編寫）」；" +
            "「我不需要寄信了」，回答：「5」；" +
            "「不用幫我寄信」，回答：「5」；" +
            "「取消寄信」，回答：「5」；"
            ));

        //緩存發送的訊息
        m_DataList.Add(new ChatGPT.SendData("user", Text));
        return m_DataList;
    }

    private List<ChatGPT.SendData> ConfirmTaskClassification(string Text)
    {
        List<ChatGPT.SendData> m_DataList = new List<ChatGPT.SendData>();
        m_DataList.Add(new ChatGPT.SendData("system",
            "我是一個郵件任務分析器，請根據用戶輸入之文句，分析用戶之目的，並根據下列要求分類任務；" +
            "若用戶在任務中有告知其他寄件、查詢訊息，請使用括弧分開並依順序紀錄：" +
            "1.郵件標題（郵件標題）" +
            "2.收件人（收件人）" +
            "3.郵件內容（郵件內容）" +
            "4.確認郵件沒問題" +
            "5.中止寄信" +

            "下列為範例：" +
            "「確認」，回答：「4」；" +
            "「信件內容沒問題」，回答：「4」；" +
            "「收件人錯誤，是富城主任」，回答：「2（富城主任）」；" +
            "「收件人有誤」，回答：「2（）」；" +
            "「郵件標題更改為會員優惠活動申訴」，回答：「1（會員優惠活動申訴）」；" +
            "「內容應該是幫我邀請福岡要不要來吃晚餐」，回答：「3（幫我邀請福岡要不要來吃晚餐）」；" +
            "「我不需要寄信了」，回答：「5」；" +
            "「不用幫我寄信」，回答：「5」；" +
            "「取消寄信」，回答：「5」；"
            ));

        //緩存發送的訊息
        m_DataList.Add(new ChatGPT.SendData("user", Text));
        return m_DataList;
    }

    private List<ChatGPT.SendData> ChangeMailBox(string Text)
    {
        List<ChatGPT.SendData> m_DataList = new List<ChatGPT.SendData>();
        m_DataList.Add(new ChatGPT.SendData("system",
            "我是一個郵件任務分析器，請根據用戶輸入之文句，分析用戶之目的，並根據下列要求分類任務；" +
            "若用戶在任務中有告知其他寄件、查詢訊息，請使用括弧分開並依順序紀錄："+
            "6.切換郵件、信箱帳戶（名稱）" +
            "5.中止切換信箱、帳戶" +

            "下列為範例：" +
            "「幫我切換到小花信箱」，回答：「6（小花）」；" +
            "「變更到富城的信箱帳號」，回答：「6（富城）」；" +
            "「幫我更換帳號」，回答：「6（）」；" +
            "「我不需要變更信箱帳戶了」，回答：「5」；" +
            "「不用幫我換帳戶」，回答：「5」；" +
            "「取消切換信箱」，回答：「5」；"
            ));

        //緩存發送的訊息
        m_DataList.Add(new ChatGPT.SendData("user", Text));
        return m_DataList;
    }

    //GPT訊息 回傳動作 (任務辨識)
    private void CallBack_M_First(string _callback)
    {
        inputChat.print(_callback);
        try
        {
            int task = int.Parse(Regex.Replace(_callback, @"\D", string.Empty));
            inputChat.print("task:" + MailData.task);

            switch(task)
            {
                case 1:
                    AiSpack($"好的。您現在正在使用 {MailBox.Name} 的信箱，正在幫您搜尋信件中...");

                    string text1 = _callback.Substring(1);
                    text1 = Regex.Replace(text1, @"[\(,（,）,\)]", string.Empty);
                    inputChat.StartCoroutine(PythonScript.Search(
                        inputChat.CallBack_G,
                        "gmail",
                        chatGPT.getApiKey(),
                        inputChat.zapier_Key,
                        $"Search {text1}"));
                    break;
                case 2:
                    if (MailData.task == -1)
                        MailData.task = task;
                    string text = _callback.Substring(1);
                    text = Regex.Replace(text, @"[\(,（,）,\)]", string.Empty);
                    inputChat.print("Replace:" + text);
                    String[] data = text.Split('；');

                    MailData.subject = data[0];
                    MailData.addressee = data[1];
                    MailData.mailText = data[2];

                    inputChat.print("subject:" + MailData.subject);
                    inputChat.print("addressee:" + MailData.addressee);
                    inputChat.print("mailText:" + MailData.mailText);

                    AiSpack($"您現在正在使用 {MailBox.Name} 的信箱。");
                    if (!String.IsNullOrEmpty(data[1]))
                    {
                        searchEmail(data[1]);
                    }

                    sendMail();
                    break;
                case 6:
                    if (MailData.task == -1)
                        guideTask = task;

                    string text2 = _callback.Substring(1);
                    text1 = Regex.Replace(text2, @"[\(,（,）,\)]", string.Empty);

                    if (String.IsNullOrEmpty(text1))
                    {
                        threadLocker = false;
                        AiSpack($"您現在正在使用 {MailBox.Name} 的信箱，請問要更換到哪一個信箱呢？");
                        break;
                    }

                    AiSpack($"好的。正在幫您切換信箱中...");
                    searchMailbox(text1);

                    break;
                case 5:
                    inputChat.CancelMailMode();
                    break;
            }
        }catch{
            AiSpack(_callback);
            threadLocker = false;
        }
    }

    //GPT訊息 回傳動作 (任務辨識)
    private void CallBack_M_Replenish(string _callback)
    {
        inputChat.print(_callback);
        try
        {
            string text = _callback.Substring(1);
            text = Regex.Replace(text, @"[\(,（,）,\)]", string.Empty);
            int task = int.Parse(Regex.Replace(_callback, @"\D", string.Empty));

            if (task == 2)
            {
                searchEmail(text);
            }

            switch (task)
            {
                case 1:
                    MailData.subject = text;
                    sendMail();
                    break;
                case 2:
                    MailData.addressee = text;
                    MailData.addressee_Email = String.Empty;
                    sendMail();
                    break;
                case 3:
                    MailData.mailText = text;
                    sendMail();
                    break;
                case 4:
                    //寄出郵件
                    inputChat.StartCoroutine(PythonScript.Search(
                    inputChat.CallBack_G,
                    "gmail",
                    chatGPT.getApiKey(),
                    inputChat.zapier_Key,
                    $"寄送一封信件給{MailData.addressee_Email}；信件標題：{MailData.subject}；信件內容：{MailData.mailText}"));
                    break;
                case 5:
                    inputChat.CancelSendMail();
                    break;

            }
        }
        catch
        {
            AiSpack($"我不太明白你的要求，也許你能說得更清楚一些。");
            threadLocker = false;
        }
    }

    private void CallBack_M_ChangeMailBox(string _callback)
    {
        inputChat.print(_callback);
        try
        {
            string text = _callback.Substring(1);
            text = Regex.Replace(text, @"[\(,（,）,\)]", string.Empty);
            int task = int.Parse(Regex.Replace(_callback, @"\D", string.Empty));

            switch (task)
            {
                case 6:
                    if (String.IsNullOrEmpty(text))
                    {
                        threadLocker = false;
                        AiSpack($"請問要更換到哪一個信箱呢？");
                        break;
                    }

                    AiSpack($"好的。正在幫您切換信箱中...");
                    searchMailbox(text);

                    break;
                case 5:
                    inputChat.CancelChangeMailBox();
                    break;

            }
        }
        catch
        {
            AiSpack($"我不太明白你的要求，也許你能說得更清楚一些。");
            threadLocker = false;
        }
    }

    void searchMailbox(string Name)
    {
        inputChat.StartCoroutine(PythonScript.SearchEmail(
            CallBack_M,
            "MailDataBase",
            Name,
            chatGPT.getApiKey(),
            $"{Name}的信箱"
            ));
    }

    void searchEmail(string addressee)
    {
        inputChat.StartCoroutine(PythonScript.SearchEmail(
            CallBack_E,
            "DataBase",
            addressee,
            chatGPT.getApiKey(),
            $"{addressee}的Email"
            ));
    }

    //GPT訊息 回傳動作 (Email查詢)
    void CallBack_E(string _callback, string addressee)
    {
        inputChat.print(_callback);
        Match match = Regex.Match(_callback, @"[a-zA-Z0-9_\.-]{1,}@[a-zA-Z0-9][\w\.-]*\.[a-zA-Z][a-zA-Z\.]*[a-zA-Z]");
        if (match.Success)
        {
            inputChat.print(match.Value);
            MailData.addressee_Email = match.Value;
            sendMail();
        }
        else
        {
            AiSpack($"通訊錄中找不到 {addressee} 的Email，請確認有無正確登錄至通訊錄。");
            MailData.addressee = String.Empty;
            threadLocker = false;
        }
    }

    //GPT訊息 回傳動作 (MailBox查詢)
    void CallBack_M(string _callback, string Name)
    {
        inputChat.print(_callback);
        Match match = Regex.Match(_callback, @"[\(,（]\S+[）,\)]");
        string indexStr = Regex.Replace(Regex.Match(_callback, @"編號\d+：").Value, @"[編號,：]", String.Empty);
        inputChat.print("indexStr:"+indexStr);
        if (match.Success && !String.IsNullOrEmpty(indexStr))
        {
            string key = Regex.Replace(match.Value, @"[(編號\d+：),\(,（,）,\)]", String.Empty);
            AiSpack($"已切換至 {Name} 的信箱。");
            inputChat.print($"index:{indexStr},key:{key}");
            inputChat.zapier_Key = match.Value;
            File.WriteAllText("MailSelect.txt", indexStr);
            threadLocker = false;
            inputChat.SendMailDone();
        }
        else
        {
            AiSpack($"找不到 {Name} 的信箱。");
            threadLocker = false;
            inputChat.SendMailDone();
        }
    }

    void sendMail()
    {

        if (string.IsNullOrEmpty(MailData.addressee))
        {
            guideTask = 1;
            AiSpack("請問這封信件是要寄送給誰呢?");
            threadLocker = false;
        }
        else if (string.IsNullOrEmpty(MailData.addressee_Email))
        {
            AiSpack("正在查詢通訊錄中...");
        }
        else if (string.IsNullOrEmpty(MailData.mailText))
        {
            guideTask = 1;
            AiSpack("請問信件內容是甚麼呢?");
            threadLocker = false;
        }
        else if (string.IsNullOrEmpty(MailData.subject))
        {
            guideTask = 1;
            AiSpack("請問信件標題由您編寫嗎，或是由我直接節錄信件內容?");
            threadLocker = false;
        }
        else
        {
            guideTask = 2;
            AiSpack(
                "以下為寄送的郵件內容請您核對：\n" +
                "標題："+ MailData.subject + "\n" +
                "收件人：" + MailData.addressee + "（" + MailData.addressee_Email + ")" +"\n" +
                "內容：" + MailData.mailText + "\n" +
                $"使用 {MailBox.Name} 的信箱寄送。"
                );
            threadLocker = false;
        }
    }

    void AiSpack(string text)
    {
        inputChat.speak(text);
        //建構對話條
        var vChatWindow = inputChat.chatWindow.transform.localPosition;
        var itemGround = inputChat.Instantiate(inputChat.GptChatItem, vChatWindow, Quaternion.identity);
        itemGround.transform.parent = inputChat.chatWindow.transform;
        itemGround.transform.GetChild(0).transform.GetChild(0).GetComponent<UnityEngine.UI.Text>().text = text;
        inputChat.checkChatsBoxToDelete();
    }

    void UserSpack(string text)
    {
        //建構對話條
        var vChatWindow = inputChat.chatWindow.transform.localPosition;
        var itemGround = inputChat.Instantiate(inputChat.UserChatItem, vChatWindow, Quaternion.identity);
        itemGround.transform.parent = inputChat.chatWindow.transform;
        itemGround.transform.GetChild(1).transform.GetChild(0).GetComponent<UnityEngine.UI.Text>().text = text;
        inputChat.checkChatsBoxToDelete();
    }
}
