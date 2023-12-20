using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

class Light_control
{
    public async Task Switch()
    {
        var info = await Info();
        string url = "http://192.168.2.3:8081/zeroconf/switch";
        string jsonBodyoff = @"{""deviceid"":""1000b47c38"",""data"":{""switch"":""off""}}";
        string jsonBodyon = @"{""deviceid"":""1000b47c38"",""data"":{""switch"":""on""}}";

        string jsonBody;
        if(info["switch"].ToString() == "on"){
            jsonBody = jsonBodyoff;
        }
        else{
            jsonBody = jsonBodyon;
        }
        Console.WriteLine(info["switch"]);
        using (HttpClient client = new HttpClient())
        {
            StringContent content = new StringContent(jsonBody, Encoding.UTF8, "application/json");

            HttpResponseMessage response = await client.PostAsync(url, content);

            if (response.IsSuccessStatusCode)
            {
                string responseContent = await response.Content.ReadAsStringAsync();
                Console.WriteLine(responseContent);
            }
            else
            {
                Console.WriteLine("請求失敗，狀態碼：" + response.StatusCode);
            }
        }
    }

   public async Task<Dictionary<string, object>> Info()
{
    string url = "http://192.168.2.3:8081/zeroconf/info";
    string jsonBody = @"{""deviceid"":""1000b47c38"",""data"":{}}";

    using (HttpClient client = new HttpClient())
    {
        StringContent content = new StringContent(jsonBody, Encoding.UTF8, "application/json");

        HttpResponseMessage response = await client.PostAsync(url, content);

        if (response.IsSuccessStatusCode)
        {
            string responseContent = await response.Content.ReadAsStringAsync();
            Console.WriteLine(responseContent);
            var infoObject = JsonConvert.DeserializeObject<Dictionary<string, object>>(responseContent);
            var data = (JObject)infoObject["data"];
            Dictionary<string, object> info = data.ToObject<Dictionary<string, object>>();
            return info;
        }
        else
        {
            Console.WriteLine("請求失敗，狀態碼：" + response.StatusCode);
        }
    }

    return null;
}



}
