using UnityEngine;
using DG.Tweening;
using UnityEngine.UI;
using System;
using System.IO;
using TMPro;


public class FadeInOnCondition : MonoBehaviour
{
    public float fadeInTime = 2.0f;
    public float fadeOutTime = 2.0f; // 文本淡出時間（秒）
    public GameObject panel;
    public TextMeshProUGUI textMeshPro;
    private bool ready = false;
    private string readyPath = Application.platform == RuntimePlatform.WindowsPlayer || Application.platform == RuntimePlatform.WindowsEditor ? @"Assets\Python\speechRecognition\ready.txt" : @"Assets/Python/speechRecognition/ready.txt";

    private void Start()
    {
        textMeshPro.alpha = 0f;

        // 開始淡入文本
        DOTween.To(() => textMeshPro.alpha, alpha => textMeshPro.alpha = alpha, 1f, fadeInTime);
           
    }

    private void Update()
    {
       
        
        if (!ready)
        {
            try
            {
                var str = File.ReadAllText(readyPath);
                if (!String.IsNullOrEmpty(str))
                {
                    ready = true;
                    panel.GetComponent<Image>().DOFade(0f, fadeInTime)
                        .OnComplete(() =>
                        {
                            panel.SetActive(false);
                        });
                    DOTween.To(() => textMeshPro.alpha, alpha => textMeshPro.alpha = alpha, 0f, fadeOutTime);
                   

                }
            }
            catch { }
            return;
        }


    }
}
