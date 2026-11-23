module DVCHModel
    use precision
    implicit none
    private
    public :: DVCHInteraction, DVCHBackground, DVCHMatterFraction

    !
    ! Sector de FONDO de DVCH: la materia no escala como (1+z)^3, sino que
    ! obedece la ecuacion de conservacion CDM-vacio del paper, y el vacio
    ! sigue la cerradura implicita
    !
    !     U(z) = x(z)^n * (1+beta) / (1 + beta*E2),   x = Om(z)/Om0
    !     E2   = Or0*(1+z)^4 + Om0*x + OL0*U
    !
    ! La ODE de conservacion se integra con RK4 en u = ln(1+z) desde z=0
    ! (donde x=1 por definicion) hasta u_max, UNA VEZ por modelo, dentro de
    ! una seccion critica de OpenMP (CAMB es paralelo sobre k). Las
    ! consultas posteriores son de solo lectura (interpolacion cubica de
    ! ln x en u + cerradura algebraica) y por tanto seguras en paralelo.
    !
    ! CONVENCION DE SALIDA: dvch_x = rho_m(a) / (rho_m0 * a**-3), es decir,
    ! la densidad de materia RELATIVA a la escala estandar a^-3.
    !
    ! Las om_* que se pasan son densidades relativas actuales
    ! (adimensionales, respecto a grhocrit).
    !

    real(dl), save :: om_m0_s, om_l0_s, om_r0_s, n_s, beta_s
    real(dl), save :: zc_s = 1.e6_dl, p_s = 10._dl
    logical, save :: switch_read = .false.
    real(dl), save, allocatable :: u_tab(:), lnx_tab(:)
    integer, save :: n_tab = 0
    real(dl), parameter :: du_tab = 0.02_dl
    real(dl), parameter :: u_max = 26._dl   ! z ~ 2e11: suficiente para CAMB
    logical, save :: built = .false.

contains

    subroutine read_switch()
        character(len=32) :: val
        if (switch_read) return
        call get_environment_variable('DVCH_ZC', val)
        if (len_trim(val) > 0) read(val, *) zc_s
        call get_environment_variable('DVCH_P', val)
        if (len_trim(val) > 0) read(val, *) p_s
        switch_read = .true.
        write(*, *) '[DVCH] switch desacoplamiento: z_c = ', zc_s, ' p = ', p_s
    end subroutine read_switch

    function switch_S(u) result(S)
        real(dl), intent(in) :: u
        real(dl) :: S, uc
        uc = log(1._dl + zc_s)
        S = 1._dl / (1._dl + exp(p_s * (u - uc)))
        if (S > 1._dl) S = 1._dl
        if (S < 0._dl) S = 0._dl
    end function switch_S

    !----------------------------------------------------------
    ! Cerradura algebraica: dado x (materia relativa a Om0) y u=ln(1+z),
    ! devuelve e2 = E^2 y vac = Om_vac(z) (autoconsistente).
    !----------------------------------------------------------
    subroutine dvch_closure(x, u, e2, vac)
        real(dl), intent(in) :: x, u
        real(dl), intent(out) :: e2, vac
        real(dl) :: rad, base, A, B, bq, disc, S_v, e2_new, vac_raw, A2
        integer :: it

        call read_switch()
        rad = om_r0_s * exp(4._dl * u)
        B = rad + om_m0_s * x
        base = om_l0_s * x**n_s
        if (beta_s <= 0._dl) then
            ! limite beta -> 0: cerradura sin supresion de curvatura
            e2 = B + base
            vac = base
        else
            A = base * (1._dl + beta_s)
            ! beta*e2^2 + (1 - beta*B)*e2 - (A + B) = 0  (raiz positiva)
            bq = 1._dl - beta_s * B
            disc = bq * bq + 4._dl * beta_s * (A + B)
            if (disc < 0._dl .or. x <= 0._dl) then
                e2 = B + A
            else
                e2 = (-bq + sqrt(disc)) / (2._dl * beta_s)
            end if
            vac = A / (1._dl + beta_s * e2)
        end if
        ! Variante de desacoplamiento: solo la desviacion del vacio respecto
        ! de LCDM (vac=1) se apaga a alto z; a z << z_c se recupera todo.
        S_v = switch_S(u)
        if (S_v >= 1._dl - 1e-14_dl) return
        vac = 1._dl + S_v * (vac - 1._dl)
        do it = 1, 80
            e2_new = rad + om_m0_s * x + om_l0_s * vac
            if (abs(e2_new - e2) < 1e-13_dl * max(1._dl, e2)) exit
            e2 = e2_new
            base = om_l0_s * x**n_s
            if (beta_s > 0._dl .and. x > 0._dl) then
                A2 = base * (1._dl + beta_s)
                bq = 1._dl - beta_s * (rad + om_m0_s * x)
                disc = bq * bq + 4._dl * beta_s * (A2 + rad + om_m0_s * x)
                if (disc >= 0._dl) then
                    vac_raw = A2 / (1._dl + beta_s * e2)
                else
                    vac_raw = A2
                end if
            else
                vac_raw = base
            end if
            vac = 1._dl + S_v * (vac_raw - 1._dl)
        end do
    end subroutine dvch_closure


    !----------------------------------------------------------
    ! Lado derecho de la ODE de conservacion, en u = ln(1+z):
    !   dx/du = (1+z) * dx/dz     (x = Om(z)/Om0, m = Om0*x)
    !----------------------------------------------------------
    function dxdu(x, u) result(derv)
        real(dl), intent(in) :: x, u
        real(dl) :: derv, e2, vac
        real(dl) :: dvac_dm, dvac_de2, denom, coeff, explicit, S_v

        call dvch_closure(x, u, e2, vac)
        if (.not. (e2 > 0._dl) .or. x <= 0._dl .or. om_m0_s <= 0._dl) then
            derv = -3._dl * x      ! valor seguro fuera de dominio
            return
        end if
        S_v = switch_S(u)
        dvac_dm = vac * n_s / (om_m0_s * x)
        dvac_de2 = -om_l0_s * x**n_s * (1._dl + beta_s) * beta_s / &
            (1._dl + beta_s * e2)**2
        dvac_dm = 1._dl + S_v * (dvac_dm - 1._dl)
        dvac_de2 = S_v * dvac_de2
        denom = 1._dl - dvac_de2
        if (denom == 0._dl) then
            derv = -3._dl * x
            return
        end if
        coeff = (1._dl + dvac_dm) * (1._dl + dvac_de2 / denom)
        explicit = 4._dl * om_r0_s * exp(3._dl * u)   ! 4*Or0*(1+z)^3
        ! dx/dz = [3x/(1+z) - (dvac_de2/denom)*explicit/Om0] / coeff
        derv = exp(u) * ((3._dl * x / exp(u) - (dvac_de2 / denom) * &
            explicit / om_m0_s) / coeff)
        if (.not. (abs(derv) < 1.e30_dl)) derv = -3._dl * x
    end function dxdu


    !----------------------------------------------------------
    ! Construye la tabla completa (0 <= u <= u_max) con RK4.
    ! Llamar solo dentro de la seccion critica.
    !----------------------------------------------------------
    subroutine build_table(ok)
        logical, intent(out) :: ok
        real(dl) :: u, x, h, k1, k2, k3, k4, unew, xnew
        integer :: n_total, i

        ok = .true.
        n_total = int(u_max / du_tab) + 1
        if (allocated(u_tab)) deallocate(u_tab, lnx_tab)
        allocate(u_tab(n_total + 1), lnx_tab(n_total + 1))
        u_tab(1) = 0._dl          ! z = 0
        lnx_tab(1) = 0._dl        ! x = 1 por definicion
        n_tab = 1
        u = 0._dl
        x = 1._dl
        do i = 2, n_total + 1
            h = du_tab
            k1 = dxdu(x, u)
            k2 = dxdu(x + 0.5_dl*h*k1, u + 0.5_dl*h)
            k3 = dxdu(x + 0.5_dl*h*k2, u + 0.5_dl*h)
            k4 = dxdu(x + h*k3, u + h)
            xnew = x + (h/6._dl)*(k1 + 2._dl*k2 + 2._dl*k3 + k4)
            unew = u + h
            if (.not. (xnew > 0._dl) .or. xnew > 1.e120_dl .or. &
                .not. (abs(xnew) < 1.e300_dl)) then
                ok = .false.
                exit
            end if
            n_tab = i
            u_tab(i) = unew
            lnx_tab(i) = log(xnew)
            u = unew
            x = xnew
        end do
        if (.not. ok) then
            n_tab = 1
            u_tab(1) = 0._dl
            lnx_tab(1) = 0._dl
        end if
    end subroutine build_table


    !----------------------------------------------------------
    ! Interpolacion cubica de ln x en u (4 puntos).
    !----------------------------------------------------------
    function interp_lnx(u) result(y)
        real(dl), intent(in) :: u
        real(dl) :: y
        integer :: j, i0
        real(dl) :: t, a0, a1, a2, a3

        if (u <= u_tab(1)) then
            y = lnx_tab(1)
            return
        end if
        if (u >= u_tab(n_tab)) then
            y = lnx_tab(n_tab)
            return
        end if
        j = 1
        do while (j + 8 < n_tab .and. u_tab(j + 8) <= u)
            j = j + 8
        end do
        do while (u_tab(j + 1) < u)
            j = j + 1
            if (j >= n_tab) exit
        end do
        i0 = min(max(j - 1, 1), n_tab - 3)
        t = (u - u_tab(i0)) / (u_tab(i0 + 1) - u_tab(i0))
        a0 = lnx_tab(i0)
        a1 = lnx_tab(i0 + 1) - lnx_tab(i0)
        a2 = lnx_tab(i0 + 2) - 2._dl*lnx_tab(i0 + 1) + lnx_tab(i0)
        a3 = lnx_tab(i0 + 3) - 3._dl*lnx_tab(i0 + 2) + 3._dl*lnx_tab(i0 + 1) - lnx_tab(i0)
        y = a0 + a1*t + a2*t*t + a3*t*t*t/3._dl
    end function interp_lnx


    !----------------------------------------------------------
    ! API publica: devuelve
    !   dvch_x  = rho_m(a) / (rho_m0 * a^-3)   (relativo a la escala LCDM)
    !   u_frac  = Om_vac(a) / Om_vac0          (cerradura implicita)
    !   e2      = E^2(a)                       (de la cerradura)
    !----------------------------------------------------------
    subroutine DVCHBackground(a, om_m0, om_l0, om_r0, n, beta, dvch_x, u_frac, e2, ok)
        real(dl), intent(in) :: a, om_m0, om_l0, om_r0, n, beta
        real(dl), intent(out) :: dvch_x, u_frac, e2
        logical, optional, intent(out) :: ok
        real(dl) :: u, lnx, x, vac
        logical :: ok_local, same_model

        if (present(ok)) ok = .true.
        if (a <= 0._dl) then
            ! Limite de radiacion dominada (a = 0 o invalido): la materia
            ! escala como a^-3 y el vacio es negligible.
            dvch_x = 0._dl
            u_frac = 0._dl
            e2 = 1.e300_dl
            return
        end if
        if (om_m0 <= 0._dl .or. om_l0 < 0._dl) then
            dvch_x = 1._dl
            u_frac = 1._dl
            e2 = 1._dl
            if (present(ok)) ok = .false.
            return
        end if

        same_model = built .and. n_tab > 4 .and. &
            abs(om_m0 - om_m0_s) < 1e-10_dl .and. &
            abs(om_l0 - om_l0_s) < 1e-10_dl .and. &
            abs(om_r0 - om_r0_s) < 1e-10_dl .and. &
            abs(n - n_s) < 1e-12_dl .and. abs(beta - beta_s) < 1e-12_dl
        if (.not. same_model) then
            !$omp critical (dvch_build)
            ! doble chequeo dentro de la seccion critica (otro hilo pudo
            ! construir la tabla mientras esperabamos)
            same_model = built .and. n_tab > 4 .and. &
                abs(om_m0 - om_m0_s) < 1e-10_dl .and. &
                abs(om_l0 - om_l0_s) < 1e-10_dl .and. &
                abs(om_r0 - om_r0_s) < 1e-10_dl .and. &
                abs(n - n_s) < 1e-12_dl .and. abs(beta - beta_s) < 1e-12_dl
            if (.not. same_model) then
                om_m0_s = om_m0
                om_l0_s = om_l0
                om_r0_s = om_r0
                n_s = n
                beta_s = beta
                call build_table(ok_local)
                if (ok_local) then
                    built = .true.
                else
                    built = .false.
                end if
            else
                ok_local = .true.
            end if
            !$omp end critical (dvch_build)
            if (.not. ok_local) then
                dvch_x = 1._dl
                u_frac = 1._dl
                e2 = 1._dl
                if (present(ok)) ok = .false.
                return
            end if
        end if

        u = log(1._dl / a)          ! ln(1+z) con z = 1/a - 1
        if (u < 0._dl) u = 0._dl    ! a > 1 (no ocurre): clamping
        if (u > u_max) then
            ! fuera de la tabla: asintota (x ~ (1+z)^3, vacio negligible)
            dvch_x = 1._dl
            u_frac = exp(3._dl*n*u) * (1._dl + beta) / &
                (1._dl + beta * (om_r0*exp(4._dl*u) + om_m0*exp(3._dl*u)))
            e2 = om_r0*exp(4._dl*u) + om_m0*exp(3._dl*u)
            return
        end if
        lnx = interp_lnx(u)
        x = exp(lnx)
        dvch_x = x * a**3           ! relativo a la escala a^-3
        call dvch_closure(x, u, e2, vac)
        if (om_l0_s > 0._dl) then
            u_frac = vac / om_l0_s
        else
            u_frac = 0._dl
        end if
    end subroutine DVCHBackground


    !----------------------------------------------------------
    ! Wrapper: solo la fraccion de materia relativa (para grho_no_de).
    !----------------------------------------------------------
    function DVCHMatterFraction(a, om_m0, om_l0, om_r0, n, beta) result(dvch_x)
        real(dl), intent(in) :: a, om_m0, om_l0, om_r0, n, beta
        real(dl) :: dvch_x, u_frac, e2

        call DVCHBackground(a, om_m0, om_l0, om_r0, n, beta, dvch_x, u_frac, e2)
    end function DVCHMatterFraction


    subroutine DVCHInteraction(E, omega_m, omega_lambda, omega_r, delta_m, &
        delta_E, n, beta, qtilde, delta_qtilde)
        real(dl), intent(in) :: E, omega_m, omega_lambda, omega_r
        real(dl), intent(in) :: delta_m, delta_E, n, beta
        real(dl), intent(out) :: qtilde, delta_qtilde
        real(dl) :: denominator, bracket, c_m, c_lambda, c_E

        if (E <= 0._dl .or. omega_m <= 0._dl .or. omega_lambda <= 0._dl) then
            qtilde = 0._dl
            delta_qtilde = 0._dl
            return
        end if

        denominator = 1._dl + n*omega_lambda/omega_m
        bracket = n - beta*(4._dl*omega_r + 3._dl*omega_m) / &
            (3._dl*(1._dl + beta*E*E))
        qtilde = -E*omega_lambda*bracket/denominator

        c_m = E*omega_lambda*beta / ((1._dl + beta*E*E)*denominator) - &
            n*E*omega_lambda*omega_lambda*bracket / &
            (omega_m*omega_m*denominator*denominator)
        c_lambda = -E*bracket/(denominator*denominator)
        c_E = -omega_lambda/denominator * &
            (bracket + 2._dl*beta*beta*E*E*(4._dl*omega_r + 3._dl*omega_m) / &
            (3._dl*(1._dl + beta*E*E)**2))

        ! The adopted vacuum-rest-frame closure sets delta Omega_Lambda=0.
        delta_qtilde = c_m*omega_m*delta_m + c_lambda*0._dl + c_E*delta_E
    end subroutine DVCHInteraction

end module DVCHModel
