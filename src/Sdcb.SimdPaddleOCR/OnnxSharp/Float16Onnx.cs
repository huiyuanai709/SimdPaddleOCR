using System.Buffers.Binary;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>
/// PP-OCRv4 server seal weights are stored as FLOAT16 so the package stays under
/// GitHub's 100 MB blob limit. Sdcb.SimdOnnx reads float initializers only, so
/// those tensors are widened to float32 before <see cref="Model"/> hands the
/// bytes to the engine. Graphs with no FLOAT16 initializers are left untouched.
/// </summary>
internal static class Float16Onnx
{
    internal static float SingleFromFloat16Bits(ushort bits)
    {
        int sign = (bits & 0x8000) << 16;
        int exponent = (bits >> 10) & 0x1F;
        int mantissa = bits & 0x3FF;
        int value;
        if (exponent == 0)
        {
            if (mantissa == 0) value = sign;
            else
            {
                exponent = -14;
                while ((mantissa & 0x400) == 0)
                {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x3FF;
                value = sign | ((exponent + 127) << 23) | (mantissa << 13);
            }
        }
        else if (exponent == 31)
            value = sign | 0x7F800000 | (mantissa << 13);
        else
            value = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        return BitConverterCompat.Int32BitsToSingle(value);
    }

    /// <summary>A new buffer when any graph initializer is FLOAT16; otherwise null.</summary>
    internal static byte[]? Widen(ReadOnlySpan<byte> source)
    {
        if (!HasFloat16Initializer(source)) return null;
        using MemoryStream output = new(source.Length + source.Length / 2);
        bool changed = false;
        RewriteMessage(source, MessageKind.Model, output, ref changed);
        return changed ? output.ToArray() : null;
    }

    private enum MessageKind { Model, Graph, Tensor }

    private static bool HasFloat16Initializer(ReadOnlySpan<byte> source)
    {
        int i = 0;
        while (i < source.Length)
        {
            Field field = ReadField(source, i);
            if (field.Number == 7 && field.Wire == 2)
            {
                ReadOnlySpan<byte> graph = source.Slice(field.PayloadStart, field.PayloadLength);
                int g = 0;
                while (g < graph.Length)
                {
                    Field initializer = ReadField(graph, g);
                    if (initializer.Number == 5 && initializer.Wire == 2 &&
                        TensorDataType(graph.Slice(initializer.PayloadStart, initializer.PayloadLength)) == 10)
                        return true;
                    g = initializer.End;
                }
            }
            i = field.End;
        }
        return false;
    }

    private static int TensorDataType(ReadOnlySpan<byte> tensor)
    {
        int i = 0;
        while (i < tensor.Length)
        {
            Field field = ReadField(tensor, i);
            if (field.Number == 2 && field.Wire == 0) return (int)field.Varint;
            i = field.End;
        }
        return 0;
    }

    private static void RewriteMessage(ReadOnlySpan<byte> message, MessageKind kind, Stream output, ref bool changed)
    {
        int i = 0;
        while (i < message.Length)
        {
            Field field = ReadField(message, i);
            if (kind == MessageKind.Model && field.Number == 7 && field.Wire == 2)
            {
                using MemoryStream graph = new();
                RewriteMessage(message.Slice(field.PayloadStart, field.PayloadLength), MessageKind.Graph, graph, ref changed);
                WriteLengthDelimited(output, 7, graph.GetBuffer(), (int)graph.Length);
            }
            else if (kind == MessageKind.Graph && field.Number == 5 && field.Wire == 2)
            {
                ReadOnlySpan<byte> tensor = message.Slice(field.PayloadStart, field.PayloadLength);
                if (TensorDataType(tensor) == 10)
                {
                    changed = true;
                    using MemoryStream widened = new();
                    RewriteTensor(tensor, widened);
                    WriteLengthDelimited(output, 5, widened.GetBuffer(), (int)widened.Length);
                }
                else
                    Write(output, message.Slice(field.Start, field.End - field.Start));
            }
            else
                Write(output, message.Slice(field.Start, field.End - field.Start));
            i = field.End;
        }
    }

    private static void RewriteTensor(ReadOnlySpan<byte> tensor, Stream output)
    {
        bool sawRaw = false;
        int i = 0;
        while (i < tensor.Length)
        {
            Field field = ReadField(tensor, i);
            if (field.Number == 2 && field.Wire == 0)
                WriteVarintField(output, 2, 1);
            else if (field.Number == 9 && field.Wire == 2)
            {
                sawRaw = true;
                ReadOnlySpan<byte> raw = tensor.Slice(field.PayloadStart, field.PayloadLength);
                if ((raw.Length & 1) != 0)
                    throw new InvalidDataException("FLOAT16 initializers must be stored in raw_data.");
                byte[] data = new byte[raw.Length * 2];
                for (int b = 0; b < raw.Length; b += 2)
                {
                    float value = SingleFromFloat16Bits(BinaryPrimitives.ReadUInt16LittleEndian(raw.Slice(b, 2)));
                    BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(b * 2, 4), BitConverterCompat.SingleToInt32Bits(value));
                }
                WriteLengthDelimited(output, 9, data, data.Length);
            }
            else
                Write(output, tensor.Slice(field.Start, field.End - field.Start));
            i = field.End;
        }
        if (!sawRaw)
            throw new NotSupportedException("FLOAT16 initializers must be stored in raw_data.");
    }

    private static void WriteVarintField(Stream output, int field, ulong value)
    {
        WriteVarint(output, (ulong)(field << 3));
        WriteVarint(output, value);
    }

    private static void WriteLengthDelimited(Stream output, int field, byte[] payload, int length)
    {
        WriteVarint(output, (ulong)((field << 3) | 2));
        WriteVarint(output, (ulong)length);
        output.Write(payload, 0, length);
    }

    private static void Write(Stream output, ReadOnlySpan<byte> data)
    {
        if (data.Length == 0) return;
        byte[] rented = System.Buffers.ArrayPool<byte>.Shared.Rent(data.Length);
        try
        {
            data.CopyTo(rented);
            output.Write(rented, 0, data.Length);
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static void WriteVarint(Stream output, ulong value)
    {
        while (value >= 0x80)
        {
            output.WriteByte((byte)(value | 0x80));
            value >>= 7;
        }
        output.WriteByte((byte)value);
    }

    private readonly struct Field
    {
        public int Number { get; init; }
        public int Wire { get; init; }
        public ulong Varint { get; init; }
        public int Start { get; init; }
        public int End { get; init; }
        public int PayloadStart { get; init; }
        public int PayloadLength { get; init; }
    }

    private static Field ReadField(ReadOnlySpan<byte> buffer, int index)
    {
        int start = index;
        (ulong key, index) = ReadVarint(buffer, index);
        int number = (int)(key >> 3);
        int wire = (int)(key & 7);
        ulong varint = 0;
        int payloadStart = index;
        int payloadLength = 0;
        switch (wire)
        {
            case 0:
                (varint, index) = ReadVarint(buffer, index);
                break;
            case 1:
                payloadLength = 8;
                index += 8;
                break;
            case 2:
                (ulong length, int after) = ReadVarint(buffer, index);
                index = after;
                payloadStart = index;
                payloadLength = checked((int)length);
                index += payloadLength;
                break;
            case 5:
                payloadLength = 4;
                index += 4;
                break;
            default:
                throw new InvalidDataException($"ONNX wire type {wire} is not supported.");
        }
        if ((uint)index > (uint)buffer.Length)
            throw new InvalidDataException("ONNX field runs past the end of the buffer.");
        return new Field
        {
            Number = number,
            Wire = wire,
            Varint = varint,
            Start = start,
            End = index,
            PayloadStart = payloadStart,
            PayloadLength = payloadLength,
        };
    }

    private static (ulong Value, int Next) ReadVarint(ReadOnlySpan<byte> buffer, int index)
    {
        ulong value = 0;
        int shift = 0;
        while (index < buffer.Length && shift < 64)
        {
            byte b = buffer[index++];
            value |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return (value, index);
            shift += 7;
        }
        throw new InvalidDataException("ONNX varint is truncated.");
    }
}
