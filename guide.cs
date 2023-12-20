using UnityEngine;
using UnityEngine.UI;
using DG.Tweening;

public class guide : MonoBehaviour
{
    public GameObject[] panels; // 存所有面板
    private int currentPanelIndex = 0;

    private void Start()
    {
        for (int i = 1; i < panels.Length; i++)
        {
            panels[i].SetActive(false); // 隐藏所有面板，除了第一個
        }
    }

    public void ShowNextPanel()
    {
        if (currentPanelIndex < panels.Length - 1)
        {
            panels[currentPanelIndex].SetActive(false);

            // 顯示下一個面板
            currentPanelIndex++;
            panels[currentPanelIndex].SetActive(true);
        }
    }

    public void ShowPreviousPanel()
    {
        if (currentPanelIndex > 0)
        {
            panels[currentPanelIndex].SetActive(false);

            // 顯示前一個面板
            currentPanelIndex--;
            panels[currentPanelIndex].SetActive(true);
        }
    }
}
