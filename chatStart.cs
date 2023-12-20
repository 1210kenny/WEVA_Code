using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using static UnityEngine.RuleTile.TilingRuleOutput;

public class chatStart : MonoBehaviour
{
    //對話窗口
    public GameObject chatWindow;
    //對話條
    public GameObject GptChatItem;
    //public AnimationControl animationControl;
    public Text chatItem;
    //用於控制動作、表情：
    //key：是否有變化, body_switch：動作編號, face_switch：表情編號
    public bool animation_key = true;
    public int animation_body_switch = 2, animation_face_switch = 2;
    [SerializeField]
    private AnimationControl animationControl;    
    // Start is called before the first frame update
    void Start()
    {
        print("start");
        // 設置畫面幀數
        Application.targetFrameRate = 60;
        // 取得對話窗口位置
        var vChatWindow = chatWindow.transform.localPosition;
        // 建構對話條
        var itemGround = Instantiate(GptChatItem, vChatWindow, Quaternion.identity);
        // 對話條插入至對話窗口
        itemGround.transform.SetParent(chatWindow.transform);
        itemGround.transform.GetChild(0).transform.GetChild(0).GetComponent<Text>().text = "你好! 有甚麼我幫的上的嗎?";
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}