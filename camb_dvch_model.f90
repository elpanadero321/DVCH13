module DVCHModel
    use precision
    implicit none
    private
    public :: DVCHInteraction

contains

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

        delta_qtilde = c_m*omega_m*delta_m + c_lambda*0._dl + c_E*delta_E
    end subroutine DVCHInteraction

end module DVCHModel
