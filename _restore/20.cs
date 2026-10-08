        }
        for (int i = 0; i < off.Length; i++)
            if (off[i] >= 0) off[i] = newOff[i];
        im2colOff = newIm2col;

        static void Release(List<(long Off, long Len)> free, long o, long len)
        {
            int k = 0;
            while (k < free.Count && free[k].Off < o) k++;
            free.Insert(k, (o, len));
            if (k + 1 < free.Count && free[k].Off + free[k].Len == free[k + 1].Off)
            {
                free[k] = (free[k].Off, free[k].Len + free[k + 1].Len);
                free.RemoveAt(k + 1);
            }
            if (k > 0 && free[k - 1].Off + free[k - 1].Len == free[k].Off)
            {
                free[k - 1] = (free[k - 1].Off, free[k - 1].Len + free[k].Len);
                free.RemoveAt(k);
            }
        }
    }

    private static byte[] Pcu(params uint[] v)
    {
        byte[] b = new byte[v.Length * 4];
        for (int i = 0; i < v.Length; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(b.AsSpan(i * 4), v[i]);
        return b;
    }

    private static ushort U16(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadUInt16LittleEndian(p[o..]);
    private static int I32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadInt32LittleEndian(p[o..]);
    private static uint U32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadUInt32LittleEndian(p[o..]);
    private static float F32(ReadOnlySpan<byte> p, int o) => BinaryPrimitives.ReadSingleLittleEndian(p[o..]);
    // fp16 bits of alpha (lo16) packed for shaders that read aux/flags via unpackHalf2x16
    private static ushort Half2Bits(float v) => BitConverter.HalfToUInt16Bits((Half)v);
    private static uint HsAux(ReadOnlySpan<byte> p) =>
        (uint)Half2Bits(F32(p, 4)) | ((uint)Half2Bits(F32(p, 8)) << 16);

}
