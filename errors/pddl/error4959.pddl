(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(clear b3)
		(on-table b5)
		(clear b5)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b9 b0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b4 b6)
		(in-tower b4)
		(on b3 b4)
		(in-tower b3)
		(green b1)
		(maroon b1)
		(red b5)
		(blue b2)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (green ?x)) (exists (?y) (and (maroon ?y) (on ?x ?y))))) (and (not (on b5 b8)) (not (on b5 b6)) (not (on b5 b4)) (not (on b5 b3)))))
)