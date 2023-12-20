using System;
using System.IO;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using Unity.VisualScripting.FullSerializer;
using UnityEngine;
using System.Text.RegularExpressions;

public class PythonScript
{

    private static string translaterPath = 
        Application.platform == RuntimePlatform.WindowsPlayer || 
        Application.platform == RuntimePlatform.WindowsEditor ?
        @"C:python.exe" : 
//        @"/opt/homebrew/Caskroom/miniforge/base/envs/M1Max/bin/python3";
        @"/Users/wangyikai/opt/anaconda3/bin/python3.9";
    //python腳本資料夾
    //private static string basePath = @"Assets\Python\";
    private static string basePath = 
        Application.platform == RuntimePlatform.WindowsPlayer || 
        Application.platform == RuntimePlatform.WindowsEditor ? 
        @"Assets\Python\" : 
        @"Assets/Python/";

    private static string outputPath =
        Application.platform == RuntimePlatform.WindowsPlayer ||
        Application.platform == RuntimePlatform.WindowsEditor ?
        @"Assets\Python\pythonOutput.txt" :
        @"Assets/Python/pythonOutput.txt";
    static string ConvertWindowsToMacOSPath(string windowsPath)
    {
        if (Path.DirectorySeparatorChar == '/' && 
            !(Application.platform == RuntimePlatform.WindowsPlayer || 
            Application.platform == RuntimePlatform.WindowsEditor)
        ){
            string fullPath = Path.GetFullPath(windowsPath);
            string macOSPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);
            return macOSPath;
        }else return windowsPath;
    }

    // Unity 調用 Python
    // 
    public static IEnumerator Search(
        System.Action<string> _callback,
        //異步回傳函式
        string programName,
        //chatGPT.getApiKey()
        //serpapi_Key
        params string[] argvs            //給 python 的其他參數
    )
    {
        string pyScriptPath;
        //string pyScriptData = "";
        if (programName == "Search")
        {
            pyScriptPath = basePath + "Search.py";
        }
        else if (programName == "gmail")
        {
            pyScriptPath = basePath + "gmail.py";
        }
        else
        {
            UnityEngine.Debug.LogError("沒有檔案.");
            yield break;
        }

        // 判斷是否有參數
        if (argvs != null)
        {
            // 添加參數
            foreach (string item in argvs)
            {
                pyScriptPath += " " + item;
            }
        }

        pyScriptPath = ConvertWindowsToMacOSPath(pyScriptPath);
        UnityEngine.Debug.Log(pyScriptPath);

        Process process = new Process();

        // ptython 的直譯器位置 python.exe
        process.StartInfo.FileName = translaterPath;
        //process.StartInfo.FileName = pyScriptPath;
        //process.StartInfo.StandardOutputEncoding = System.Text.Encoding.GetEncoding(950); //回傳正確的中文編碼
        process.StartInfo.UseShellExecute = false;
        //process.StartInfo.Arguments = pyScriptData;       // (exe用) 純參數
        process.StartInfo.Arguments = pyScriptPath;     // 路徑+參數
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.RedirectStandardInput = true;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.CreateNoWindow = true;        // 不顯示執行窗口

        string answer = "";

        // 開始執行，獲取執行輸出，添加結果輸出委託
        process.Start();
        process.BeginOutputReadLine();
        process.OutputDataReceived += new DataReceivedEventHandler(GetData);

        //等待python回傳資料
        while(string.IsNullOrEmpty(answer) == true)
        {
            UnityEngine.Debug.Log("Wait...");
            yield return new WaitForSeconds(1f);//停止1秒
        }

        _callback(answer);

        // 結果輸出委託
        void GetData(object sender, DataReceivedEventArgs e)
        {
            //輸出不為空
            if (string.IsNullOrEmpty(e.Data) == false)
            {
                //UnityEngine.Debug.Log(e.Data);
                answer = e.Data;
            }
        }

        yield return null;
    }

    // Unity 調用 Python
    // 
    public static IEnumerator SearchEmail(
        System.Action<string, string> _callback,
        //異步回傳函式
        string programName,
        string backData,
        //chatGPT.getApiKey()
        //serpapi_Key
        params string[] argvs            //給 python 的其他參數
    )
    {

        string pyScriptPath;
        if (programName == "DataBase")
        {
            pyScriptPath = basePath + "DataBase.py";
        }
        else if (programName == "MailDataBase")
        {
            pyScriptPath = basePath + "DataBase_MailDataBase.py";
        }
        else
        {
            UnityEngine.Debug.LogError("沒有檔案.");
            yield break;
        }

        // 判斷是否有參數
        if (argvs != null)
        {
            // 添加參數
            foreach (string item in argvs)
            {
                pyScriptPath += " " + item;
            }
        }
        pyScriptPath = ConvertWindowsToMacOSPath(pyScriptPath);
        UnityEngine.Debug.Log(pyScriptPath);

        Process process = new Process();

        // ptython 的直譯器位置 python.exe
        process.StartInfo.FileName = translaterPath;
        //process.StartInfo.StandardOutputEncoding = System.Text.Encoding.GetEncoding(950); //回傳正確的中文編碼
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.Arguments = pyScriptPath;     // 路徑
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.RedirectStandardInput = true;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.CreateNoWindow = true;        // 不顯示執行窗口
        string answer = "";
        // 開始執行，獲取執行輸出，添加結果輸出委託
        process.Start();
        process.BeginOutputReadLine();
        process.OutputDataReceived += new DataReceivedEventHandler(GetData);
        //等待python回傳資料
        while (string.IsNullOrEmpty(answer) == true)
        {
            UnityEngine.Debug.Log("Wait...");
            yield return new WaitForSeconds(1f);//停止1秒
        }

        _callback(answer, backData);

        // 結果輸出委託
        void GetData(object sender, DataReceivedEventArgs e)
        {
            //輸出不為空
            if (string.IsNullOrEmpty(e.Data) == false)
            {
                //UnityEngine.Debug.Log(e.Data);
                answer = e.Data;
            }
        }
        yield return null;
    }


    public static IEnumerator speechRecognition(
        System.Action<Process> _callback,
        string key,
        string region,
        params string[] argvs            //給 python 的其他參數
    )
    {
        string pyScriptPath;
        pyScriptPath = @"speechRecognition.py";

        UnityEngine.Debug.Log(pyScriptPath);

        Process process = new Process();
        string WorkingDirectorypath = @"Assets\Python\speechRecognition";
        WorkingDirectorypath = ConvertWindowsToMacOSPath(WorkingDirectorypath);
        // ptython 的直譯器位置 python.exe
        process.StartInfo.FileName = translaterPath;
        //process.StartInfo.StandardOutputEncoding = System.Text.Encoding.GetEncoding(950); //回傳正確的中文編碼
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.Arguments = $"{pyScriptPath} --key {key} --region {region}";     // 路徑
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.RedirectStandardInput = true;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.WorkingDirectory = WorkingDirectorypath;
        process.StartInfo.CreateNoWindow = true;        // 不顯示執行窗口
        // 開始執行，獲取執行輸出，添加結果輸出委託
        process.Start();
        process.BeginOutputReadLine();

        _callback(process);
        yield return null;
    }
}