      module no_model
        
        implicit none
        
        LOGICAL INIT
        SAVE INIT
        DATA INIT /.FALSE./

        ! Constants
        real(8), parameter :: SELU_ALPHA = 1.6732632423543772848170429916717d0
        real(8), parameter :: SELU_SCALE = 1.0507009873554804934193349852946d0
        real(8), parameter :: pi = 3.14159265359
        

        real(8), allocatable :: f_weights_in(:,:), f_biases_in(:)
        real(8), allocatable :: f_weights_hid(:,:,:)
        real(8), allocatable :: f_biases_hid(:,:)
        real(8), allocatable :: f_weights_out(:,:), f_biases_out(:)
        real(8) :: scale_strain
        real(8) :: scale_stress

        SAVE f_weights_in
        SAVE f_biases_in
        SAVE f_weights_hid
        SAVE f_biases_hid
        SAVE f_weights_out
        SAVE f_biases_out
        SAVE scale_strain
        SAVE scale_stress

      contains
        ! this subroutine loads the scaling parameters for strain and stress from a file
C        subroutine load_scaling_parameters(filename, 
C     1 scale_strain, scale_stress)
        subroutine load_scaling_parameters(filename)
          implicit none
          character(len=256), intent(in) :: filename
C          real(8) , intent(out) :: scale_strain
C          real(8) , intent(out) :: scale_stress
          IF (.NOT. INIT) THEN
            open(unit=10, file=filename, status="old", action="read")
            read(10,*) scale_strain
            read(10,*) scale_stress
            close(10)
          end if
        end subroutine load_scaling_parameters

        ! this subroutine loads the model parameters (weights and biases) from a file
        subroutine load_model_parameters(filename)
          implicit none
          character(len=256), intent(in) :: filename
          ! integer, intent(in) :: d, k
          integer :: input_dim, output_dim, hidden_dim, num_layers
          character(len=20) :: hidden_dim_str, input_dim_str
          character(len=20) :: out_dim_str
          integer :: i_start, i_end, comma_position
          character(len=256) :: layer_name
          character(len=256) :: in_layer_name, out_layer_name
          integer :: io_status
          integer :: i, j, layer_itr

          
          IF (.NOT. INIT) THEN
          
            open(unit=10, file=filename, status='old', iostat=io_status)
            if (io_status /= 0) then
                print *, "Error opening file: ", filename
                stop 
            end if

            ! First pass: determine model dimensions
            input_dim = 0
            hidden_dim = 0
            output_dim = 0
            num_layers = 0

            do 
              read(10, '(A)', iostat=io_status) layer_name
              if (io_status /= 0) exit

              if (index(layer_name, '.func..mlp..layers.[0]..weight') 
     1 > 0) then
                backspace(10)
                read(10, '(A)') in_layer_name
                ! Find the position of '(' and ')' to extract the size of input and hidden layers
                i_start = INDEX(in_layer_name, '(') + 1
                i_end = INDEX(in_layer_name, ')') - 1
                comma_position = INDEX(in_layer_name, ',')
                hidden_dim_str=in_layer_name(i_start:comma_position-1)
                input_dim_str=in_layer_name(comma_position+1:i_end)
                
                read(input_dim_str, *) input_dim
                read(hidden_dim_str, *) hidden_dim

                num_layers = num_layers + 1
                  
              else if (index(layer_name, 'weight') > 0) then
                backspace(10)
                read(10, '(A)') out_layer_name
                num_layers = num_layers + 1

              end if

            end do

            ! Find the position of '(' and ')' to extract the size of output layer
            i_start = INDEX(out_layer_name, '(') + 1
            i_end = INDEX(out_layer_name, ')') - 1
            comma_position = INDEX(out_layer_name, ',')
            out_dim_str=out_layer_name(i_start:comma_position-1)
            READ(out_dim_str, *) output_dim

            num_layers = num_layers - 2
          
            ! print *, "d =", d
            ! print *, "k =", k
            print *, "input_dim =", input_dim
            print *, "hidden_dim =", hidden_dim
            print *, "output_dim =", output_dim
            print *, "num_hidden_layers =", num_layers

            rewind(10)

            ! Allocate memory for input layer weights and biases of f
            if (allocated(f_weights_in)) deallocate(f_weights_in)
            if (allocated(f_biases_in)) deallocate(f_biases_in)
            if (allocated(f_weights_hid)) deallocate(f_weights_hid)
            if (allocated(f_biases_hid)) deallocate(f_biases_hid)
            if (allocated(f_weights_out)) deallocate(f_weights_out)
            if (allocated(f_biases_out)) deallocate(f_biases_out)
            
            allocate(f_weights_in(hidden_dim, input_dim))
            allocate(f_biases_in(hidden_dim))
            allocate(f_weights_hid(num_layers,hidden_dim, hidden_dim))
            allocate(f_biases_hid(num_layers,hidden_dim))
            allocate(f_weights_out(output_dim,hidden_dim))
            allocate(f_biases_out(output_dim))

            ! Second pass: read weights and biases
            ! consider that input, hidden and output layers read seperately 
            layer_itr = 1

            do
              read(10, '(A)', iostat=io_status) layer_name
              if (io_status /= 0) exit
              
              if (index(layer_name, 'layers.[0]..weight') > 0) then
                  do i = 1, hidden_dim
                    do j = 1, input_dim
                      read(10, *) f_weights_in(i, j)
                    end do
                  end do
              else if (index(layer_name, 'layers.[0]..bias') > 0) then
                  do i = 1, hidden_dim
                        read(10, *) f_biases_in(i)
                  end do
              else if (index(layer_name, 'layers') > 0 .and. 
     1 layer_itr <= num_layers) then
                if (index(layer_name, 'weight') > 0) then
                  do i = 1, hidden_dim
                    do j = 1, hidden_dim
                      read(10, *) f_weights_hid(layer_itr, i, j)
                    end do
                  end do
                else if (index(layer_name, 'bias') > 0) then
                  do i = 1, hidden_dim
                    read(10, *) f_biases_hid(layer_itr, i)
                  end do
                  layer_itr = layer_itr + 1
                end if
              else if (index(layer_name, 'layers') > 0 .and. 
     1 layer_itr > num_layers) then
                if (index(layer_name, 'weight') > 0) then
                  do i = 1, output_dim
                    do j = 1, hidden_dim
                      read(10, *) f_weights_out(i, j)
                    end do
                  end do
                else if (index(layer_name, 'bias') > 0) then
                  do i = 1, output_dim
                    read(10, *) f_biases_out(i)
                  end do
                  layer_itr = 1
                end if

              end if

            end do
          
            close(10)

            print *, "Model parameters loaded successfully."

          end if
        end subroutine load_model_parameters
        ! this is the SELU activation function
        function selu(x)
          real(8), intent(in) :: x
          real(8) :: selu
            
          if (x > 0.0d0) then
            selu = SELU_SCALE * x
          else
            selu = SELU_SCALE * SELU_ALPHA * (exp(x) - 1.0d0)
          end if
        end function selu
        

        ! this is the ReLU activation function
        function relu(x)
          real(8), intent(in) :: x
          real(8) :: relu

          if (x > 0.0d0) then
            relu = x
          else
            relu = 0.0d0
          end if
        end function relu

        subroutine construct_jacobian(output, stress_dim, k_inv, jac_matrix)
          integer, intent(in) :: stress_dim, k_inv
          real(8), intent(in) :: output(:)
          real(8), intent(out) :: jac_matrix(stress_dim+k_inv, stress_dim)

          integer :: sym_size
          integer :: i, j, idx
          real(8) :: sym_matrix(stress_dim, stress_dim)
          real(8) :: hidden_matrix(k_inv, stress_dim)

          ! sizes
          sym_size = stress_dim * (stress_dim + 1) / 2

          ! initialize
          jac_matrix = 0.0d0
          sym_matrix = 0.0d0
          hidden_matrix = 0.0d0
C
          ! ---------------------------------------
          ! 22. sym_part -> symmetric stress matrix
          ! ---------------------------------------
          ! Fill upper triangle including diagonal
          idx = 1
          do i = 1, stress_dim
              do j = i, stress_dim
                  sym_matrix(i,j) = output(idx)
                  idx = idx + 1
              end do
          end do
C
          ! Copy upper triangle to lower triangle
          do i = 2, stress_dim
              do j = 1, i-1
                  sym_matrix(i,j) = sym_matrix(j,i)
              end do
          end do
C
          jac_matrix(1:stress_dim, 1:stress_dim) = sym_matrix
C
          ! ---------------------------------------
          ! 3. hidden_part -> lower block
          ! ---------------------------------------
          do i = 1, k_inv
              do j = 1, stress_dim
                  hidden_matrix(i,j) = output(idx)
                  idx = idx + 1
              end do
          end do
C
          jac_matrix(stress_dim+1:stress_dim+k_inv, 1:stress_dim) = 
     1    hidden_matrix

        end subroutine construct_jacobian

        ! this subroutine implements the feed-forward pass of the neural network
        subroutine feed_forward_network(input, output, weights_in,
     1 biases_in, weights_hid, biases_hid, weights_out, biases_out)
          real(8), intent(in) :: input(:)
          real(8), intent(out) :: output(:)
          real(8), intent(in) :: weights_in(:,:), biases_in(:)
          real(8), intent(in) :: weights_hid(:,:,:), biases_hid(:,:)
          real(8), intent(in) :: weights_out(:,:), biases_out(:)
          real(8) :: layer_hidden(size(weights_hid,2))
          real(8) :: current_layer_hidden(size(weights_hid,2))
          integer :: i, j

          ! input layer
          do i = 1, size(weights_in, 1)
            layer_hidden(i)=selu(dot_product(input,
     1 weights_in(i,:))+ biases_in(i))
          end do

          ! hidden layer
          do i = 1, size(weights_hid, 1)
            current_layer_hidden=layer_hidden
            do j = 1, size(weights_hid, 2)
              layer_hidden(j)=selu(dot_product(current_layer_hidden,
     1 weights_hid(i,j,:))+biases_hid(i,j))
            end do
          end do

          ! output layer
          do i = 1, size(weights_out, 1)
            output(i)=dot_product(layer_hidden,
     1 weights_out(i,:))+biases_out(i)
          end do

        end subroutine feed_forward_network

        ! this is the main subroutine that computes the stress increment, state variable increment and the Jacobian matrix for the UMAT
        subroutine no_forward(STRESS, STATE, DSTRAN, DSTRESS, DDSDDE,
     1 DSTATEV, HIST, dt, d, k)
          integer, intent(in) :: d, k
          real(8), intent(in) :: DSTRAN(d), STRESS(d), dt
          real(8), intent(in) :: STATE(k)
          real(8), intent(out) :: DSTATEV(k)
          real(8), intent(out) :: DSTRESS(d)
          real(8), intent(out) :: DDSDDE(d,d), HIST(k,d)
          real(8) :: jac_matrix(d+k, d)
          real(8) :: input_f(2*d+k)
          real(8) :: input_d(d+1)
          real(8) :: output_f((((d + 1) * d) / 2) + (k * d))

          ! Compute jacobian
          input_f = [STRESS, STATE, DSTRAN/dt]
          call feed_forward_network(input_f, output_f,
     1 f_weights_in, f_biases_in, f_weights_hid,
     2 f_biases_hid, f_weights_out, f_biases_out)

          call construct_jacobian(output_f, d, k, jac_matrix)
          DDSDDE = jac_matrix(1:d,1:d)
          HIST = jac_matrix(d+1:d+k,1:d)

          DSTRESS = matmul(DDSDDE, DSTRAN)
          DSTATEV = matmul(HIST, DSTRAN)

        end subroutine no_forward
      
      end module no_model

      SUBROUTINE UEXTERNALDB(LOP,LRESTART,TIME,DTIME,
     1 KSTEP,KINC)
      USE no_model
      INCLUDE 'ABA_PARAM.INC'
      DIMENSION TIME(2)
      character(LEN=256) :: modelfilename
      character(LEN=256) :: scalefilename
      IF (LOP .EQ. 0) THEN
        WRITE(*,*) 'Loading model parameters...'
        scalefilename = 'C:\Temp\Beam\scales.dat'
        call load_scaling_parameters(scalefilename)
C
        modelfilename = 'C:\Temp\Beam\mlp_parameters.dat'
        call load_model_parameters(modelfilename)
        INIT = .TRUE.
      ENDIF
      RETURN
      END
! UMAT subroutine
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,
     2 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
C
      USE no_model
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),
     1 DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS),
     2 STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1),
     3 PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3),
     4 DTEMP(1)
C --- For perturbation check ---
      real(8) :: STRESS_n(NTENS)
      real(8) :: STATEV_n(NSTATV)
      real(8) :: STRESS_ref(NTENS)
      real(8) :: STRESS_pert(NTENS)
      real(8) :: STATEV_tmp(NSTATV)
      real(8) :: DDSDDE_tmp(NTENS,NTENS)
      real(8) :: error(NTENS,NTENS)
      real(8) :: rel(1)
      real(8) :: DSTRESS_tmp(NTENS)
      real(8) :: DSTATEV_tmp(NSTATV)
      real(8) :: DSTRAN_pert(NTENS)
      real(8) :: C_fd(NTENS,NTENS)
      real(8) :: EPS_PERT
      integer :: i, j
      integer :: NOEL, NPT, LAYER, KSPT
C
      real(8) :: strainP(NTENS), strainM(NTENS)
      real(8) :: stressP(NTENS), stressM(NTENS)
      real(8) :: stateP(NSTATV), stateM(NSTATV)
C            
      real(8) :: DSTRESS(NTENS), HIST(NSTATV,NTENS)
      real(8) :: DSTATEV(NSTATV)
C      
      character(LEN=256) :: modelfilename
      character(LEN=256) :: scalefilename

C      
      DSTRAN = (DSTRAN / scale_strain)
      STRESS = ((STRESS *1e6) / scale_stress)
      STATEV = ((STATEV *1e6) / scale_stress)
C
      STRESS_n = STRESS
      STATEV_n = STATEV
      ! this is the main function for model evaluation
      call no_forward(STRESS, STATEV, DSTRAN, DSTRESS, DDSDDE,
     1 DSTATEV, HIST, DTIME, NTENS, NSTATV)
C      
      STRESS = STRESS + DSTRESS
      STATEV = STATEV + DSTATEV
      STRESS_ref = STRESS
C       
C ==============================================================
C 3. Finite-difference perturbation tangent check (DEBUG ONLY)
C ==============================================================
    !   EPS_PERT = 1.0D-6
    !   DO j = 1, NTENS
    !      DSTRAN_pert = DSTRAN
    !      DSTRAN_pert(j) = DSTRAN_pert(j) + EPS_PERT
    !      call no_forward( STRESS_n, STATEV_n, DSTRAN_pert,
    !  1                    DSTRESS_tmp, DDSDDE_tmp,
    !  2                    DSTATEV_tmp, HIST,
    !  3                    DTIME, NTENS, NSTATV )
    !      STRESS_pert = STRESS_n + DSTRESS_tmp
    !      DO i = 1, NTENS
    !         C_fd(i,j) = (STRESS_pert(i) - STRESS_ref(i)) / (EPS_PERT)
    !      END DO
    !   END DO
C
      STRESS = (STRESS * scale_stress) /1e6
      STATEV = (STATEV * scale_stress) /1e6 

      DDSDDE = ((scale_stress / scale_strain)/1e6 ) * DDSDDE
C
      RETURN
      END