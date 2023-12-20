using UnityEngine;
using UnityEngine.UI;
using DG.Tweening;

public class MovePanel : MonoBehaviour
{
    public RectTransform panelToMove;
    public float moveDuration = 1.0f;
    public Vector2 startPosition;
    public Vector2 targetPosition = new Vector2(326f, 0f); 
    private bool isPanelMoved = false;
    public Button moveButton;
    public GameObject text1;
    public GameObject text2;

    private void Start()
    {
        // 紀錄初始位置
        startPosition = panelToMove.anchoredPosition;
    }

    public void MovePanelOnClick()
    {
        if (!isPanelMoved)
        {
            // 移到目標位置
            panelToMove.DOAnchorPos(targetPosition, moveDuration).OnComplete(() =>
            {
                isPanelMoved = true;
                text1.SetActive(true);
                text2.SetActive(false);
            });
        }
        else
        {
            // 回到初始位置
            panelToMove.DOAnchorPos(startPosition, moveDuration).OnComplete(() =>
            {
                isPanelMoved = false;
                text1.SetActive(false);
                text2.SetActive(true);
            });
        }
    }
}
