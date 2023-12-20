using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DG.Tweening;

public class canvasmove : MonoBehaviour
{
    public CanvasGroup canvasGroup;
    public float showDuration = 1.0f;
    public float hideDuration = 1.0f;

    private void Start()
    {
        canvasGroup = GetComponent<CanvasGroup>();
        canvasGroup.alpha = 0;
        canvasGroup.blocksRaycasts = false;
        // 檢查是否已經設置過應用程序的"IsFirstTime"標誌
        if (!PlayerPrefs.HasKey("IsFirstTime"))
        {
            Debug.Log("使用者第一次使用");

            canvasGroup.DOFade(1, showDuration);
            canvasGroup.blocksRaycasts = true;

            // 設置"IsFirstTime"標誌為true，以標記使用者已經使用過一次
            PlayerPrefs.SetInt("IsFirstTime", 1);
            PlayerPrefs.Save(); // 保存PlayerPrefs
        }
        else
        {
            Debug.Log("使用者不是第一次使用");
        }
    }

    public void ShowCanvas()
    {
        canvasGroup.DOFade(1, showDuration);
        canvasGroup.blocksRaycasts = true; 
    }

    public void HideCanvas()
    {
        canvasGroup.DOFade(0, hideDuration);
        canvasGroup.blocksRaycasts = false; 
    }
}