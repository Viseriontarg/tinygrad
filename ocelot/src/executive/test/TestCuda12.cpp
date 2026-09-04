#include <ocelot/ir/PTXOperand.h>
#include <ocelot/parser/PTXParser.h>
#include <hydrazine/ArgumentParser.h>
#include <hydrazine/Test.h>

#include <cmath>
#include <cstring>
#include <vector>

extern "C" void ptx_run(const char*, int, void**, int, int, int, int, int, int, int);

class TestCuda12: public test::Test {
	void run(const std::string& body, void* output, void* input, unsigned int threads = 1, unsigned int blocks = 1) {
		const std::string source = R"(
.version 8.8
.target sm_86
.address_size 64
.visible .entry test(.param .u64 output, .param .u64 input) {
.reg .b64 %rd<8>;
.reg .b32 %r<8>;
.reg .b16 %h<4>;
.reg .f32 %f<4>;
.reg .f64 %d<4>;
.reg .pred %p<2>;
ld.param.u64 %rd0, [output];
ld.param.u64 %rd1, [input];
mov.u32 %r0, %ctaid.x;
mov.u32 %r1, %ntid.x;
mov.u32 %r2, %tid.x;
mad.lo.u32 %r0, %r0, %r1, %r2;
)" + body + "\nret;\n}";
		void* args[] = {output, input};
		ptx_run(source.c_str(), 2, args, threads, 1, 1, blocks, 1, 1, 0);
	}

	bool testHalfRounding() {
		const struct { double value; unsigned short rn, rz, rm, rp; } cases[] = {
			{0.0, 0, 0, 0, 0}, {-0.0, 0x8000, 0x8000, 0x8000, 0x8000},
			{1.0, 0x3c00, 0x3c00, 0x3c00, 0x3c00},
			{1.00048828125, 0x3c00, 0x3c00, 0x3c00, 0x3c01},
			{1.00146484375, 0x3c02, 0x3c01, 0x3c01, 0x3c02},
			{-1.00048828125, 0xbc00, 0xbc00, 0xbc01, 0xbc00},
			{-1.00146484375, 0xbc02, 0xbc01, 0xbc02, 0xbc01},
			{0x1p-24, 1, 1, 1, 1}, {0x1p-25, 0, 0, 0, 1}, {-0x1p-25, 0x8000, 0x8000, 0x8001, 0x8000},
			{0x1p-100, 0, 0, 0, 1}, {-0x1p-100, 0x8000, 0x8000, 0x8001, 0x8000},
			{65504, 0x7bff, 0x7bff, 0x7bff, 0x7bff}, {65520, 0x7c00, 0x7bff, 0x7bff, 0x7c00},
			{-65520, 0xfc00, 0xfbff, 0xfc00, 0xfbff}, {1e100, 0x7c00, 0x7bff, 0x7bff, 0x7c00},
			{-1e100, 0xfc00, 0xfbff, 0xfc00, 0xfbff}, {INFINITY, 0x7c00, 0x7c00, 0x7c00, 0x7c00},
			{-INFINITY, 0xfc00, 0xfc00, 0xfc00, 0xfc00}, {NAN, 0x7e00, 0x7e00, 0x7e00, 0x7e00}
		};
		std::vector<double> input;
		for (const auto& c : cases) input.push_back(c.value);
		for (const std::string mode : {"rn", "rz", "rm", "rp"}) {
			std::vector<unsigned short> output(input.size());
			run(R"(
mul.wide.u32 %rd2, %r0, 8;
add.u64 %rd2, %rd1, %rd2;
ld.global.f64 %d0, [%rd2];
cvt.)" + mode + R"(.f16.f64 %h0, %d0;
mul.wide.u32 %rd2, %r0, 2;
add.u64 %rd2, %rd0, %rd2;
st.global.u16 [%rd2], %h0;
)", output.data(), input.data(), input.size());
			for (unsigned int i = 0; i < input.size(); ++i) {
				const auto expected = mode == "rn" ? cases[i].rn : mode == "rz" ? cases[i].rz : mode == "rm" ? cases[i].rm : cases[i].rp;
				if (output[i] != expected) {
					status << "cvt." << mode << ".f16.f64 " << input[i] << ": expected " << expected << ", got " << output[i] << "\n";
					return false;
				}
			}
		}
		return true;
	}

	bool testHalfValues() {
		struct Result { float value; unsigned short bits; };
		static_assert(sizeof(Result) == 8, "PTX output stride");
		std::vector<unsigned short> input(65536);
		std::vector<Result> output(input.size());
		for (unsigned int i = 0; i < input.size(); ++i) input[i] = i;
		run(R"(
mul.wide.u32 %rd2, %r0, 2;
add.u64 %rd3, %rd1, %rd2;
ld.global.u16 %h0, [%rd3];
cvt.f32.f16 %f0, %h0;
cvt.rn.f16.f32 %h1, %f0;
mul.wide.u32 %rd2, %r0, 8;
add.u64 %rd3, %rd0, %rd2;
st.global.f32 [%rd3], %f0;
st.global.u16 [%rd3+4], %h1;
)", output.data(), input.data(), 256, 256);
		for (unsigned int i = 0; i < input.size(); ++i) {
			const bool nan = (i & 0x7c00) == 0x7c00 && (i & 0x3ff);
			_Float16 native;
			std::memcpy(&native, &input[i], sizeof(native));
			const float expected = native;
			if ((nan ? !std::isnan(output[i].value) : std::memcmp(&output[i].value, &expected, sizeof(expected)) != 0) ||
				output[i].bits != (nan ? 0x7e00 : i)) {
				status << "half conversion failed for bits " << i << ": " << output[i].value << ", " << output[i].bits << "\n";
				return false;
			}
		}
		return true;
	}

	bool testBFloatFma() {
		unsigned short output = 0, input[] = {0x3fc0, 0x3f81, 0x8080};
		// 1.5 * 1.0078125 is a BF16 tie. The tiny negative addend must break it downward.
		run(R"(
ld.global.u16 %h0, [%rd1];
ld.global.u16 %h1, [%rd1+2];
ld.global.u16 %h2, [%rd1+4];
fma.rn.bf16 %h3, %h0, %h1, %h2;
st.global.u16 [%rd0], %h3;
)", &output, input);
		if (output != 0x3fc1) { status << "BF16 fused rounding: " << output << "\n"; return false; }
		input[1] = 0x3f83;
		input[2] = 0x0080;
		run(R"(
ld.global.u16 %h0, [%rd1];
ld.global.u16 %h1, [%rd1+2];
ld.global.u16 %h2, [%rd1+4];
fma.rn.bf16 %h3, %h0, %h1, %h2;
st.global.u16 [%rd0], %h3;
)", &output, input);
		if (output != 0x3fc5) { status << "BF16 fused rounding: " << output << "\n"; return false; }
		return true;
	}

	bool testPackedHalf() {
		unsigned int output = 0, input[] = {0x40003c00, 0x44004200, 0x46004500};
		run(R"(
ld.global.u32 %r1, [%rd1];
ld.global.u32 %r2, [%rd1+4];
ld.global.u32 %r3, [%rd1+8];
fma.rn.f16x2 %r4, %r1, %r2, %r3;
st.global.u32 [%rd0], %r4;
)", &output, input);
		if (output != 0x4b004800) { status << "packed half fma: " << output << "\n"; return false; }
		unsigned int zeros[2] = {};
		run(R"(
mov.b32 %r1, 0x80000000;
mov.b32 %r2, 0x00008000;
min.f16x2 %r3, %r1, %r2;
max.f16x2 %r4, %r1, %r2;
st.global.u32 [%rd0], %r3;
st.global.u32 [%rd0+4], %r4;
)", zeros, zeros);
		if (zeros[0] != 0x80008000 || zeros[1] != 0) { status << "half min/max signed zeros\n"; return false; }
		return true;
	}

	bool testBitOperands() {
		unsigned short halfResults[4] = {};
		run(R"(
.reg .f16 half;
mov.b16 half, 0xbc00;
mov.b16 %h0, half;
st.global.u16 [%rd0+6], %h0;
mov.b16 %h0, 0x3c00;
set.eq.f16.f16 %h1, %h0, %h0;
st.global.u16 [%rd0], %h1;
set.lt.u16.f16 %h1, %h0, %h0;
st.global.u16 [%rd0+2], %h1;
neg.bf16 %h1, %h0;
st.global.u16 [%rd0+4], %h1;
)", halfResults, halfResults);
		if (halfResults[0] != 0x3c00 || halfResults[1] != 0 || halfResults[2] != 0xbc00 || halfResults[3] != 0xbc00) {
			status << "half comparison or BF16 negation\n";
			return false;
		}
		unsigned int comparison = 0;
		run(R"(
mov.b32 %r1, 0xbf800000;
mov.b32 %r2, 0;
set.lt.f32.f32 %r3, %r1, %r2;
st.global.u32 [%rd0], %r3;
)", &comparison, &comparison);
		if (comparison != 0x3f800000) { status << "set with bit registers: " << comparison << "\n"; return false; }
		run(R"(
bfi.b32 %r1, 0xab, 0x11223344, 8, 8;
st.global.u32 [%rd0], %r1;
)", &comparison, &comparison);
		if (comparison != 0x1122ab44) { status << "bfi.b32: " << comparison << "\n"; return false; }
		unsigned long long output = 0, input = 0x1122334455667788ull;
		run(R"(
ld.global.u64 %rd2, [%rd1];
bfi.b64 %rd3, 171, %rd2, 8, 8;
st.global.u64 [%rd0], %rd3;
)", &output, &input);
		if (output != 0x112233445566ab88ull) { status << "bfi.b64: " << output << "\n"; return false; }
		unsigned short half = 0x1234;
		unsigned int packed[2] = {};
		run(R"(
ld.global.u16 %h0, [%rd1];
mov.b32 %r1, {0, %h0};
mov.b32 %r2, {%h0, 0};
st.global.u32 [%rd0], %r1;
st.global.u32 [%rd0+4], %r2;
)", packed, &half);
		if (packed[0] != 0x12340000 || packed[1] != 0x1234) { status << "mixed register/immediate vector\n"; return false; }
		return true;
	}

	bool testInvalidOperands() {
		unsigned int output = 0;
		try { run("mov.b32 {%h0, 0}, %r0;", &output, &output); }
		catch (const parser::PTXParser::Exception&) { return true; }
		status << "accepted an immediate in a destination vector\n";
		return false;
	}

	bool doTest() {
		try { return testHalfRounding() && testHalfValues() && testBFloatFma() && testPackedHalf() && testBitOperands() && testInvalidOperands(); }
		catch (const std::exception& e) { status << e.what() << "\n"; return false; }
	}
public:
	TestCuda12() { name = "TestCuda12"; }
};

int main(int argc, char** argv) {
	hydrazine::ArgumentParser parser(argc, argv);
	TestCuda12 test;
	parser.parse("-v", test.verbose, false, "Print out info after the test.");
	parser.parse();
	test.test();
	return test.passed();
}
