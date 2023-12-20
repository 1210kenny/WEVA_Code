using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class Googlesheet : MonoBehaviour
{
    void Start()
    {
        // Start the coroutine of sending the request to the API url.
        StartCoroutine(Upload());
    }

    IEnumerator Upload()
    {
        // Create the form object.
        WWWForm form = new WWWForm();
        // Add the method data to the form object. (read or write data)
        form.AddField("method", "read");

        // Sending the request to API url with form object.
        using (UnityWebRequest www = UnityWebRequest.Post("https://script.google.com/macros/s/AKfycbwKag2Up1qryVvH6ZkqE_owbXvJahJRsetUjIBiw4rOM7rzDscy9zyHi3ii20Sj5JIrcQ/exec", form))
        {
            yield return www.SendWebRequest();

            if (www.isNetworkError || www.isHttpError)
            {
                Debug.Log(www.error);
            }
            else
            {
                // �^�Ǹ�Ʈw�Ҧ����O
                string responseText = www.downloadHandler.text;
                print(responseText);
                Debug.Log("Form upload complete!");

                // �ˬd��Ʈw���O�_��"�}�ҹq�O"�����O
                if (ContainsKeyword(responseText, "�}�ҹq�O"))
                {
                    Debug.Log("Google Sheet contains '�}�ҹq�O' data!");
                    // Perform your desired action here.

                }
                else
                {
                    Debug.Log("Google Sheet does not contain '�}�ҹq�O' data.");
                }
            }
        }
    }

    bool ContainsKeyword(string text, string keyword)
    {
        // Split the response text into rows.
        string[] rows = text.Split('\n');

        // Iterate through each row.
        for (int i = 0; i < rows.Length; i++)
        {
            // Split the row into columns.
            string[] columns = rows[i].Split(',');

            // Iterate through each column.
            for (int j = 0; j < columns.Length; j++)
            {
                // Check if the column contains the keyword.
                if (columns[j].Contains(keyword))
                {
                    return true; // Keyword found.
                }
            }
        }

        return false; // Keyword not found.
    }
}
