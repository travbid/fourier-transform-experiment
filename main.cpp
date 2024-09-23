#include <array>
#include <bit>
#include <chrono>
#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <print>
#include <ranges>
#include <string>
#include <system_error>
#include <vector>

namespace chrono = std::chrono;

namespace {

using complex = std::complex<double>;

constexpr auto pi = std::numbers::pi_v<double>;
// constexpr bool IS_LITTLE_ENDIAN = std::bit_cast<uint32_t>(std::array<uint8_t, 4>{1, 0, 0, 0}) == 1;

uint16_t operator""_u16(unsigned long long int val) {
    return val;
}
uint32_t operator""_u32(unsigned long long int val) {
    return val;
}

void check_stream(const std::fstream &stream) {
    if (stream.eof()) {
        std::println(stderr, "Unexpected end of file");
        std::exit(EXIT_FAILURE);
    }
    if (stream.bad()) {
        std::println(stderr, "Failed to read: bad");
        std::exit(EXIT_FAILURE);
    }
    if (stream.fail()) {
        std::println(stderr, "Failed to read: fail");
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
void expect(const T &actual, const T &expected) {
    if (actual != expected) {
        std::println(stderr, "Expected {}", expected);
        std::exit(EXIT_FAILURE);
    }
}

template <>
void expect<std::string>(const std::string &actual, const std::string &expected) {
    if (actual != expected) {
        std::println(stderr, "Expected \"{}\"", expected);
        std::exit(EXIT_FAILURE);
    }
}

template <size_t Size>
std::array<char, Size> read_bytes(std::fstream &stream) {
    std::array<char, Size> buf;
    stream.read(buf.data(), Size);
    check_stream(stream);
    return buf;
}

int16_t read_i16(std::fstream &stream) {
    const auto buf = read_bytes<2>(stream);
    return std::bit_cast<int16_t>(buf);
}

uint16_t read_u16(std::fstream &stream) {
    const auto buf = read_bytes<2>(stream);
    return std::bit_cast<uint16_t>(buf);
}

uint32_t read_u32(std::fstream &stream) {
    const auto buf = read_bytes<4>(stream);
    return std::bit_cast<uint32_t>(buf);
}

std::string read_string(std::fstream &stream, const size_t len) {
    std::vector<char> buf(len, 0);
    stream.read(buf.data(), buf.size());
    check_stream(stream);
    return std::string(buf.data(), buf.data() + buf.size());
}

size_t str_to_num(const std::string &str) {
    size_t num = 0;
    const auto result = std::from_chars(str.data(), str.data() + str.size(), num, 10);
    if (result.ec != std::errc{}) {
        std::println("Error parsing \"{}\": {}", str, std::make_error_code(result.ec).message());
        std::exit(EXIT_FAILURE);
    }
    if (result.ptr != str.data() + str.size()) {
        std::println("Error parsing \"{}\": invalid argument", str);
        std::exit(EXIT_FAILURE);
    }
    return num;
};

struct InfoItem {
    std::string key;
    std::string value;
};

struct WavInfo {
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    std::vector<InfoItem> info_items;
};

WavInfo read_wav_info(std::fstream &wav_stream) {
    const auto chunk_id = read_string(wav_stream, 4);
    expect<std::string>(chunk_id, "RIFF");

    const auto chunk_size = read_u32(wav_stream);
    std::println("      chunk_size: {}", chunk_size);

    const std::string wave_tag = read_string(wav_stream, 4);
    expect<std::string>(wave_tag, "WAVE");

    const std::string subchunk1_id = read_string(wav_stream, 4);
    expect<std::string>(subchunk1_id, "fmt ");

    const auto subchunk1_size = read_u32(wav_stream);
    std::println("  subchunk1_size: {}", subchunk1_size);
    expect(subchunk1_size, 16_u32);  // 16 for PCM

    const auto audio_format = read_u16(wav_stream);
    std::println("    audio_format: {}", audio_format);
    expect(audio_format, 1_u16);  // 1 for uncompressed

    const auto num_channels = read_u16(wav_stream);
    std::println("    num_channels: {}", num_channels);
    // expect(num_channels, 1_u16); // 1 for mono, 2 for stereo

    const auto sample_rate = read_u32(wav_stream);
    std::println("     sample_rate: {}", sample_rate);

    const auto byte_rate = read_u32(wav_stream);
    std::println("       byte_rate: {}", byte_rate);

    const auto block_align = read_u16(wav_stream);
    std::println("     block_align: {}", block_align);

    const auto bits_per_sample = read_u16(wav_stream);
    std::println(" bits_per_sample: {}", bits_per_sample);
    expect(bits_per_sample, 16_u16);

    std::vector<InfoItem> info_items;
    const std::string next_tag = read_string(wav_stream, 4);
    if (next_tag == "LIST") {
        const auto list_size = read_u32(wav_stream);
        std::println("       list_size: {}", list_size);

        const std::string list_type_id = read_string(wav_stream, 4);
        expect<std::string>(list_type_id, "INFO");

        size_t ix = 4;  // Length of "INFO"
        while (ix < list_size) {
            const auto key = read_string(wav_stream, 4);
            std::println("       key: {}", key);
            const auto len = read_u32(wav_stream) + 1;
            std::println("       len: {}", len);
            const auto val = read_string(wav_stream, len);
            std::println("       val: \"{}\"", val);
            ix += 4 + 4 + len;

            info_items.push_back(InfoItem{.key = key, .value = val});
        }

        const std::string subchunk2_id = read_string(wav_stream, 4);
        expect<std::string>(subchunk2_id, "data");
    } else {
        expect<std::string>(next_tag, "data");
    }

    const auto subchunk2_size = read_u32(wav_stream);
    std::println("  subchunk2_size: {}", subchunk2_size);

    const auto num_samples = subchunk2_size / (num_channels * sizeof(uint16_t));
    std::println("  num_samples: {}", num_samples);

    return WavInfo{
        .num_channels = num_channels,
        .sample_rate = sample_rate,
        .byte_rate = byte_rate,
        .block_align = block_align,
        .bits_per_sample = bits_per_sample,
        .info_items = std::move(info_items),
    };
}

struct Components {
    size_t k_bin;
    double magnitude;
    double angle;
};

struct DftInfo {
    double frequency_hz;
    double amplitude;
    double phase;
};

struct FourierReturn {
    std::vector<Components> components;
    size_t iterations;
};

FourierReturn discrete_fourier_transform(const std::vector<int16_t> &data) {
    const size_t N = data.size();
    std::vector<Components> components;
    size_t iterations = 0;
    for (size_t bin = 0; bin != N; ++bin) {
        const double k = bin;
        complex fk{0, 0};
        for (size_t n = 0; n != N; ++n) {
            ++iterations;
            const double xn = data.at(n);
            const auto fn = xn * std::polar(1.0, -(2.0 * pi * k * n) / N);
            fk += fn;
        }
        const auto magnitude = std::sqrt((fk.real() * fk.real()) + (fk.imag() * fk.imag())) / N;
        const auto angle = std::atan2(fk.imag(), fk.real());
        components.push_back(Components{.k_bin = bin, .magnitude = magnitude, .angle = angle});
    }
    return {.components = components, .iterations = iterations};
}

// x is a power of two if it has exactly 1 set bit
constexpr bool is_power_of_two(size_t x) {
    return (x & (x - 1)) == 0 and (x != 0);
}

std::vector<complex> fast_fourier_transform_inner(const std::vector<int16_t> &data, size_t &iterations) {
    ++iterations;

    const auto N = data.size();
    const auto M = data.size() / 2;

    // https://www.youtube.com/watch?v=htCj9exbGo0
    if (N != 1) {
        std::vector<int16_t> evens;
        evens.reserve(M);
        std::vector<int16_t> odds;
        odds.reserve(M);
        for (size_t i = 0; i != N; i += 2) {
            evens.push_back(data[i]);
            odds.push_back(data[i + 1]);
        }
        const auto evens_fft = fast_fourier_transform_inner(evens, iterations);
        const auto odds_fft = fast_fourier_transform_inner(odds, iterations);
        std::vector<complex> ret(N, complex(0, 0));
        for (size_t i = 0; i != M; ++i) {
            const auto w = std::polar(1.0, (2.0 * pi * i) / data.size());
            ret[i] = evens_fft[i] + (w * odds_fft[i]);
            ret[i + (ret.size() / 2)] = evens_fft[i] - (w * odds_fft[i]);
        }
        return ret;
    }

    const double xn = data[0];
    // m = 0, therefore e^(-2*pi*k*m/N) = 1
    // const auto exp = -(2.0 * pi * k * m) / (N / depth);
    // const auto fn = xn * std::polar(1.0, exp);
    // return fn;
    return {complex(xn, 0)};
}

FourierReturn fast_fourier_transform(const std::vector<int16_t> &data) {
    if (not is_power_of_two(data.size())) {
        const auto new_size = std::bit_ceil(data.size());
        auto data_copy = data;
        data_copy.insert(data_copy.cend(), new_size - data.size(), 0);
        return fast_fourier_transform(data_copy);
    }

    size_t iterations = 0;
    const auto fft = fast_fourier_transform_inner(data, iterations);

    const double N = data.size();
    std::vector<Components> components;
    for (size_t bin = 0; bin != fft.size(); ++bin) {
        const auto fk = fft[bin];
        const auto magnitude = std::sqrt((fk.real() * fk.real()) + (fk.imag() * fk.imag())) / N;
        const auto angle = std::atan2(fk.imag(), fk.real());
        components.push_back(Components{.k_bin = bin, .magnitude = magnitude, .angle = angle});
    }
    return {.components = components, .iterations = iterations};
}

}  // namespace

int main(const int argc, char const *const *argv) {
    if (argc != 5) {
        std::println(
            stderr, "Usage: {} <path/to/input/file.wav> <samples offset> <number of samples> <path/to/output/file.csv>",
            argv[0]);
        return EXIT_FAILURE;
    }

    const std::string in_filename{argv[1]};
    const std::string offset_str{argv[2]};
    const std::string num_samples_str{argv[3]};
    const std::string out_filename{argv[4]};

    // Open input file
    auto wav_stream = std::fstream(in_filename, std::ios_base::in | std::ios_base::binary);
    const auto &[num_channels, sample_rate, byte_rate, block_align, bits_per_sample, info_items] =
        read_wav_info(wav_stream);

    const size_t byte_offset = str_to_num(offset_str) * sizeof(int16_t) * num_channels;
    const size_t num_samples = str_to_num(num_samples_str);

    // Seek to offset
    wav_stream.seekg(byte_offset, std::ios_base::cur);

    // Read selected samples
    std::vector<int16_t> channel_data;
    for (size_t i = 0; i != num_samples; ++i) {
        const auto sample = read_i16(wav_stream);
        channel_data.push_back(sample);
        wav_stream.seekg((num_channels - 1) * sizeof(uint16_t), std::ios_base::cur);
    }
    std::println("");

    const auto dft = [&channel_data] {
        const auto t0 = chrono::steady_clock::now();
        const auto dft = discrete_fourier_transform(channel_data);
        const auto t1 = chrono::steady_clock::now();
        std::println("DFT took {}µs ({} iterations)", chrono::duration_cast<chrono::microseconds>(t1 - t0).count(),
                     dft.iterations);
        return dft.components;
    }();

    const auto fft = [&channel_data] {
        const auto t0 = chrono::steady_clock::now();
        const auto fft = fast_fourier_transform(channel_data);
        const auto t1 = chrono::steady_clock::now();
        std::println("FFT took {}µs ({} iterations)", chrono::duration_cast<chrono::microseconds>(t1 - t0).count(),
                     fft.iterations);
        return fft.components;
    }();

    // Check DFT and FFT results match
    if (dft.size() != fft.size()) {
        std::println("Expected equal sizes for DFT and FFT");
        std::println("DFT: {}, FFT: {}", dft.size(), fft.size());
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i != dft.size(); ++i) {
        if (std::abs(dft[i].magnitude - fft[i].magnitude) > 0.125) {
            std::println("DFT and FFT mismatch at index {}", i);
            std::println("DFT: {}, FFT: {}", dft[i].magnitude, fft[i].magnitude);
            return EXIT_FAILURE;
        }
    }

    // Transform bins to frequencies
    const auto freq_range = dft | std::views::transform([N = channel_data.size(), sample_rate](const Components &comp) {
                                return DftInfo{.frequency_hz = static_cast<double>(comp.k_bin) * sample_rate / N,
                                               .amplitude = comp.magnitude,
                                               .phase = comp.angle};
                            });

    // Write output CSV
    std::fstream out(out_filename, std::ios::out | std::ios::binary);
    {
        std::string labels{"Frequency,Amplitude,Phase\n"};
        out.write(labels.c_str(), labels.size());
    }
    for (const auto &[freq, amp, phase] : freq_range) {
        const auto csv_line = std::format("{},{},{}\n", freq, amp, phase);
        out.write(csv_line.c_str(), csv_line.size());
    }

    std::println("Done");
}
