using UnityEngine;
using UnityEngine.UI;
using System.IO;


public class ShowAssistent : MonoBehaviour {
    public Text NowAssistent;
    private void Start() {
        string AssistentName;
        try
        {
            AssistentName = File.ReadAllText("NowAssistent.txt");
        }
        catch
        {
            // 如果讀取文件失敗，創建一個新的文件
            using (var fs = File.Create("NowAssistent.txt"))
            {
                fs.Close(); // 確保文件被關閉和釋放
            }
            AssistentName = "米拉(Mira)";
            File.WriteAllText("NowAssistent.txt", AssistentName);
        }
        NowAssistent.text = "Now Assistant : " + AssistentName;
    }
}