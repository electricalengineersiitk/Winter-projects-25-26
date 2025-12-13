// AXI - Advanced Xtensible Interface  -> memory management like ram(sram,bram)
// write and read data 
// address 
// mem block (RAM) <- read_addr
//                 -> read_data  
//                 <- wrie_addr
//                 <- write_data
//                 <- write_valid

// AXI - general interface to a RAM -> Bursts 
// Handshaking 

// AXI             G1 read_addr
//                 G2 read_data  
//                 G3 wrie_addr
//                 G4 write_data
//                 G5 write_valid


//                 1.) READ BURSTS 
//                 0,4,8 ---- 36 } 10 contiguous 4 byte packets -> incremental burst --> 4 bytes/transfer
//                 G1 read_addr -> ar_ready
//                              <- ar_valid 
//                              <- ar_burst - 2'b01
//                              <- ar_addr
//                              <- ar_len = 3'b010(no. of transfers)
//                              <- ar_size = 4 = 2 ^ ar_len

//                 G2 read_data <- r_ready 
//                              -> r_valid
//                              -> read_data
//                              -> r_response
//                              -> r_last 
                             

//                2.) Write Bursts 
//                 G3 write_addr <- aw_burst --2'b01
//                              <- aw_addr 
//                              <- aw_len (no. of transfers) 
//                              <- aw_size =2^aw_len
//                              <- aw_valid
//                              -> aw_ready

//                 G4 write_data <- write_data
//                               <- w_strb --> data strobing (0xffffffff) -> AXI lite protocol
//                               <- w_last 
//                               -> w_ready 
//                               <- w_valid 

//                 G5 write_response -> b_response
//                                   <- b_ready
//                                   -> b_valid
                
`timescale 1ns / 1ps

module axi_lite_slave_example #(
    parameter C_AXI_DATA_WIDTH = 32,
    parameter C_AXI_ADDR_WIDTH = 4
)
(
    // Global Signals
    input wire  CLK,
    input wire  RESETN,

    // Write Address Channel (AW)
    input wire  [C_AXI_ADDR_WIDTH-1:0]  S_AXI_AWADDR,
    input wire                          S_AXI_AWVALID,
    output reg                          S_AXI_AWREADY,

    // Write Data Channel (W)
    input wire  [C_AXI_DATA_WIDTH-1:0]  S_AXI_WDATA,
    input wire  [C_AXI_DATA_WIDTH/8-1:0] S_AXI_WSTRB,
    input wire                          S_AXI_WVALID,
    output reg                          S_AXI_WREADY,

    // Write Response Channel (B)
    output reg  [1:0]                   S_AXI_BRESP,
    output reg                          S_AXI_BVALID,
    input wire                          S_AXI_BREADY,

    // Read Address Channel (AR)
    input wire  [C_AXI_ADDR_WIDTH-1:0]  S_AXI_ARADDR,
    input wire                          S_AXI_ARVALID,
    output reg                          S_AXI_ARREADY,

    // Read Data Channel (R)
    output reg  [C_AXI_DATA_WIDTH-1:0]  S_AXI_RDATA,
    output reg  [1:0]                   S_AXI_RRESP,
    output reg                          S_AXI_RVALID,
    input wire                          S_AXI_RREADY,

    // User Logic Interfaces (Example Registers)
    output reg [C_AXI_DATA_WIDTH-1:0] slv_reg0, // Read/Write Register at 0x00
    input wire [C_AXI_DATA_WIDTH-1:0] ro_reg // Read-Only Input Register (e.g., status) at 0x04
);

// --- Internal Signals and Registers ---
reg [C_AXI_ADDR_WIDTH-1:0] axi_awaddr_reg;
reg [C_AXI_ADDR_WIDTH-1:0] axi_araddr_reg;

// Register 1: Read/Write at 0x00
// Register 2: Read-Only (ro_reg) at 0x04

// Local parameters for register addressing
localparam ADDR_REG0 = 4'h0; // Address 0x00
localparam ADDR_REG1 = 4'h4; // Address 0x04 // read only
//read handshake 
always@(posedge CLK)
begin
    if(!RESETN)begin
        S_AXI_ARREADY <=1'b0;
        axi_araddr_reg<= 1'b0;

    end
    else begin
        S_AXI_ARREADY <= S_AXI_ARREADY | (S_AXI_ARVALID & S_AXI_RREADY & S_AXI_BREADY) 
        if(S_AXI_ARREADY == 1'b0 && S_AXI_ARVALID=1'b1)
        S_AXI_ARREADY <=1'b1;
        axi_araddr_reg <= S_AXI_ARREADY;
    end
end
// write handshake 
always@ posedge( CLK)
begin 
    if(!RESETN)
    begin
        S_AXI_AWREADY <=1'b0;
        axi_awaddr_reg<='b0;
    end
    else  begin 
        S_AXI_AWREADY <= S_AXI_AWREADY | (S_AXI_AWVALID & S_AXI_WREADY & S_AXI_BREADY) 
        if(S_AXI_AWREADY == 1'b0 && S_AXI_AWVALID=1'b1)begin 
        S_AXI_AWREADY<=1'b1;
        axi_awaddr_reg<=S_AXI_AWADDR;
    end
end
end



// write logic
if(S_AXI_AWREADY && S_AXI_WREADY && S_AXI_WVALID && S_AXI_AWVALID) begin
 S_AXI_BVALID <= 1'b1;
 S_AXI_BRESP <= 2'b00;

 case(axi_awaddr_reg[C_AXI_ADDR_WIDTH-1:2])
 ADDR_REG0[C_AXI_ADDR_WIDTH-1:2] begin
    slv_reg0<= S_AXI_WDATA;
 end
 ADDR_REG1[C_AXI_ADDR_WIDTH-1:2] begin 
    S_AXI_BRESP <=2'b10; // slave error 
 end
 default: begin 
    S_AXI_BRESP <= 2'b10;
end
 endcase 
end

// write 
always@(posedge CLK) begin 
 if(!RESETN) begin 
    S_AXI_ARVALID<=1'b0;
    S_AXI_BRESP <=1'b0;
    slv_reg0 <=1'b0;

end
 // deassert after handshake 
else begin
    if(S_AXI_BREADY && S_AXI_BVALID)
    S_AXI_BVALID<=1'b0;
end
// checking for a succesful write operation 
if( S_AXI_AWREADY && S_AXI_WREADY && S_AXI_AWVALID && S_AXI_WVALID)
begin 
 S_AXI_BVALID <=1'b1;
 S_AXI_BRESP <=2'b00;
 case(axi_awaddr_reg[C_AXI_ADDR_WIDTH-1:2])

 ADDR_REG0[C_AXI_ADDR_WIDTH-1:2] begin 
    slv_reg0 <= S_AXI_WDATA;
 end
 ADDR_REG1[C_AXI_ADDR_WIDTH-1:2] begin 
    S_AXI_BRESP<=2'b10;
 end
 default : S_AXI_BRESP<=2'b10;
 endcase

 S_AXI_AWREADY <= 1'b0;
 S_AXI_WREADY <=1'b0;
end
end

// read logic 

always @(posedge CLK )
begin
    if(!RESETN) begin
        S_AXI_RREADY <=1'b0;
        S_AXI_RVALID<=1'b0;
        S_AXI_RRESP<=2'b00;
     end
      // deassert after handshake 
     else begin 
        if(S_AXI_RREADY && S_AXI_RVALID)
        begin
            S_AXI_RREADY <=1'b0;
            S_AXI_ARREADY<=1'b0; 
        end
     end

     // checking for a succesful read operation 
     if(S_AXI_ARREADY && S_AXI_ARVALID && S_AXI_RVALID==1'b0 )
     begin
          S_AXI_RVALID<=1'b1;
          S_AXI_RRESP<=2'b00;

          case(axi_awaddr_reg[C_AXI_ADDR_WIDTH-1:2])
          ADDR_REG0[C_AXI_ADDR_WIDTH-1:2]begin 
            S_AXI_RDATA<=slv_reg0;
            end
            ADDR_REG1[C_AXI_ADDR_WIDTH-1:2] begin
                S_AXI_RDATA<=ro_reg;
            end
            default : begin 
                S_AXI_RDATA<='b0;
                S_AXI_RRESP<=2'b10;
            end
          endcase
     end
end

endmodule
