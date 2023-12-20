using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Runtime.InteropServices;

public class ChineseProgram
{
    private const int LocaleSystemDefault = 0x0800;
    private const int LcmapSimplifiedChinese = 0x02000000;
    private const int LcmapTraditionaChinese = 0x04000000;

    [DllImport("kernel32", CharSet = CharSet.Auto, SetLastError = true)]
    private static extern int LCMapString(int locale, int dwMapFlags, string lpSrcStr, int cchSrc,
                                          [Out] string lpDestStr, int cchDest);
    public static string ToSimplifiedChinese(string argSource)
    {
        var t = new string(' ', argSource.Length);
        LCMapString(LocaleSystemDefault, LcmapSimplifiedChinese, argSource, argSource.Length, t, argSource.Length);
        return t;
    }

    public static string ToTraditionaChinese(string argSource)
    {
        var t = new string(' ', argSource.Length);
        LCMapString(LocaleSystemDefault, LcmapTraditionaChinese, argSource, argSource.Length, t, argSource.Length);
        return t;
    }
}